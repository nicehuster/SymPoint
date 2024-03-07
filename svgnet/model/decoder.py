

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from .position_embedding import PositionEmbeddingCoordsSine
from .basic_operators import get_subscene_features
from .dn_query import CDNQueries

class Decoder(nn.Module):
    def __init__(self,cfg,planes):
        super().__init__()    
        self.num_decoders = cfg.num_decoders
        self.num_classes = cfg.semantic_classes
        self.planes = planes
        self.dropout = cfg.dropout
        self.pre_norm = cfg.pre_norm
        self.shared_decoder = cfg.shared_decoder
        self.dim_feedforward = cfg.dim_feedforward
        self.mask_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.num_queries = cfg.num_queries
    
    
        # PARAMETRIC QUERIES
        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, cfg.hidden_dim)
        self.query_pos = nn.Embedding(self.num_queries, cfg.hidden_dim)
        
        self.mask_embed_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        self.mask_features_head = nn.Sequential(
            nn.Linear(planes[0], cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        
        self.class_embed_head = nn.Linear(cfg.hidden_dim, self.num_classes+1)
        
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()
        
        num_shared = self.num_decoders if not self.shared_decoder else 1
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=cfg.gauss_scale,
                                                       normalize=cfg.normalize_pos_enc)
        
        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, plane in enumerate(self.planes[::-1][:-1]):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                )
                tmp_squeeze_attention.append(nn.Linear(plane, self.mask_dim))

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                )
                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                        activation="gelu",
                    )
                )
            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(cfg.hidden_dim)
        
    
    def get_pos_encs(self,coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            coords_batch = coords[i]
            scene_min = coords_batch.min(dim=0)[0][None, ...]
            scene_max = coords_batch.max(dim=0)[0][None, ...]
            with autocast(enabled=False):
                tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])
                pos_encodings_pcd.append(tmp.permute(2,0,1))
        return pos_encodings_pcd
    
    def forward(self, stage_list):
        
        srcs,coords = [], []
        for stage in stage_list["up"]:
            src,coord = stage['f_out'], stage['p_out']
            srcs.append(src)
            coords.append(coord)
            
        srcs.reverse() # 
        coords.reverse() # p5,..p1
        
        
        pos_encodings_pcd = self.get_pos_encs(coords[:-1])
        mask_features = self.mask_features_head(srcs[-1])
        # PARAMETRIC QUERIES
        queries = self.query_feat.weight[:,None,:]
        query_pos = self.query_pos.weight[:,None,:]

        predictions_class = []
        predictions_mask = []
        
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
                
            total_step = len(self.planes[:-1])
            for i in range(total_step):
                src_pcd = self.lin_squeeze[decoder_counter][i](srcs[i])[:,None,:]
                output_class, outputs_mask,attn_mask = self.mask_module(queries,
                                                          mask_features,
                                                          stage_list,
                                                          total_step-i,
                                                          ret_attn_mask=True)
                
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                output = self.cross_attention[decoder_counter][i](
                    queries,
                    src_pcd,
                    memory_mask=attn_mask.transpose(0,1),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos_encodings_pcd[i],
                    query_pos=query_pos
                )
                
                output = self.self_attention[decoder_counter][i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos
                )
                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                )

                predictions_class.append(output_class.transpose(0,1))
                predictions_mask.append(outputs_mask.transpose(0,1))
                   
        output_class, outputs_mask = self.mask_module(queries,mask_features)
        
        predictions_class.append(output_class.transpose(0,1))
        predictions_mask.append(outputs_mask.transpose(0,1))
        
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class , predictions_mask
            ),
            "stage_list": stage_list,
        }
        return out


    def postprocess_for_dn(self, predictions_class, predictions_mask):
        n_lys = len(predictions_class)
        dn_predictions_class, predictions_class = [predictions_class[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                  [predictions_class[i][:, -self.num_queries:] for i in range(n_lys)]
        dn_predictions_mask, predictions_mask = [predictions_mask[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                [predictions_mask[i][:, -self.num_queries:] for i in range(n_lys)]
        return predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask


    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        
    def mask_module(self, query_feat, mask_features,stage_list=None,step=None,ret_attn_mask=False):

        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        output_masks = mask_embed @ mask_features.T
        if ret_attn_mask and step:
            attn_mask = output_masks.flatten(0,1).transpose(0,1)
            attn_mask = get_subscene_features("up", step, stage_list, attn_mask, torch.tensor([4, 4, 4, 4]))
            attn_mask = attn_mask.transpose(0,1)[:,None,:]
            attn_mask = (attn_mask.sigmoid().repeat(1,self.num_heads,1)<0.5).bool()
            attn_mask = attn_mask.detach()
            return outputs_class, output_masks,attn_mask
        else:
            return outputs_class, output_masks
        
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask= None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
         
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask= None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
