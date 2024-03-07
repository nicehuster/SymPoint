
import math,re
import os,glob,json
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from collections import defaultdict
import numpy as np
#from sklearn.metrics.pairwise import euclidean_distances

LABEL_NUM = 35
COMMANDS = ['Line', 'Arc','circle', 'ellipse']
import mmcv, argparse

def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split', type=str, default="test",
                        help='the split of dataset')
    parser.add_argument('--data_dir', type=str, default="./dataset/test/test/svg_gt",
                        help='save the downloaded data')
    args = parser.parse_args()
    return args

def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3]
    minx, miny, width, height = [int(float(x)) for x in root.attrib['viewBox'].split(' ')]
    
    commands = []
    args = [] # (x1,y1,x2,y2,x3,y3,x4,y4) 4points
    lengths = []
    semanticIds = []
    instanceIds = []
    strokes = []
    layerIds = []
    widths = []
    inst_infos = defaultdict(list)
    id = 0
    for g in root.iter(ns + 'g'):
        id +=1
        # path
        for path in g.iter(ns + 'path'):
            try:
                path_repre = parse_path(path.attrib['d'])
            except Exception as e:
                raise RuntimeError("Parse path failed!{}, {}".format(svg_file, path.attrib['d']))
            
            path_type = path_repre[0].__class__.__name__
            commands.append(COMMANDS.index(path_type))
            length = path_repre.length()
            lengths.append(length)
            layerIds.append(id)
            semanticId = int(path.attrib['semanticId']) - 1 if 'semanticId' in path.attrib else LABEL_NUM
            instanceId = int(path.attrib['instanceId']) if 'instanceId' in path.attrib else -1
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            rgb = list(map(int,re.findall(r'\d+',path.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(path.attrib["stroke-width"])])
            inds = [0, 1/3, 2/3, 1.0]
            arg = []
            for ind in inds:
                point = path_repre.point(ind)
                arg.extend([point.real,point.imag])
            args.append(arg)
            inst_infos[(instanceId,semanticId)].extend(arg)
            
        
        # circle
        for circle in g.iter(ns + 'circle'):
             
            cx = float(circle.attrib['cx'])
            cy = float(circle.attrib['cy'])
            r = float(circle.attrib['r'])
            semanticId = int(circle.attrib['semanticId']) - 1 if 'semanticId' in circle.attrib else LABEL_NUM
            instanceId = int(circle.attrib['instanceId']) if 'instanceId' in circle.attrib else -1
            circle_len = 2 * math.pi * r
            lengths.append(circle_len)
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            commands.append(COMMANDS.index("circle"))
            layerIds.append(id)
            rgb = list(map(int,re.findall(r'\d+',circle.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(circle.attrib["stroke-width"])])
            thetas = [0,math.pi/2, math.pi, 3 * math.pi/2,]
            arg = []
            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x,y])
            args.append(arg)
            inst_infos[(instanceId,semanticId)].extend(arg)
               
        # ellipse
        for ellipse in g.iter(ns + 'ellipse'):
            cx = float(ellipse.attrib['cx'])
            cy = float(ellipse.attrib['cy'])
            rx = float(ellipse.attrib['rx'])
            ry = float(ellipse.attrib['ry'])
            
            semanticId = int(ellipse.attrib['semanticId']) - 1 if 'semanticId' in ellipse.attrib else LABEL_NUM
            instanceId = int(ellipse.attrib['instanceId']) if 'instanceId' in ellipse.attrib else -1
            if rx>ry: 
                a,b = rx, ry
            else:
                a,b = ry, rx
            ellipse_len = 2* math.pi *b + 4*(a - b)
            lengths.append(ellipse_len)
            commands.append(COMMANDS.index("ellipse"))
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            layerIds.append(id)
            rgb = list(map(int,re.findall(r'\d+',ellipse.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(ellipse.attrib["stroke-width"])])
            thetas = [0,math.pi/2, math.pi, 3 * math.pi/2,]
            arg = []
            for theta in thetas:
                x, y = cx + a * math.cos(theta), cy + b * math.sin(theta)
                arg.extend([x,y])
            args.append(arg)
            inst_infos[(instanceId,semanticId)].extend(arg)
            
        
            
    assert len(args) == len(lengths) ,'error'
    assert len(semanticIds) ==  len(instanceIds), 'error'
    obj_cts = []
    obj_boxes = []
    for (inst_id, sem_id),coords in inst_infos.items():
        if inst_id<0: continue
        coords = np.array(coords).reshape(-1,2)
        x1,y1 = np.min(coords[:,0]), np.min(coords[:,1])
        x2,y2 = np.max(coords[:,0]), np.max(coords[:,1])
        obj_cts.append([(x1+x2)/2,(y1+y2)/2,0,inst_id])
        obj_boxes.append([x1,y1,x2,y2,sem_id])
    
    coords = np.array(args).reshape(-1,4,2)
   
    json_dicts = {
        "commands":commands,
        "args":args,
        "lengths":lengths,
        "semanticIds":semanticIds,
        "instanceIds":instanceIds,
        "width":width,
        "height":height,
        "obj_cts": obj_cts, #(x,y,z)
        "boxes": obj_boxes,
        "rgb": strokes,
        "layerIds":layerIds,
        "widths": widths
    }
    return json_dicts

def save_json(json_dicts,out_json):
    json.dump(json_dicts, open(out_json, 'w'), indent=4)
    
def process(data):
    
    svg_file, save_dir = data
    json_dicts = parse_svg(svg_file)
    filename = svg_file.split("/")[-1].replace(".svg",".json")
    out_json = os.path.join(save_dir,filename)
    save_json(json_dicts,out_json)

if __name__=="__main__":
    
    args = parse_args()
    svg_paths = glob.glob(os.path.join(args.data_dir,'*.svg'))
    
    save_dir = os.path.join("./dataset/",args.split, "jsons")
    os.makedirs(save_dir,exist_ok=True)
    
    inputs = []
    for svg_path in svg_paths: inputs.append([svg_path, save_dir])

    mmcv.track_parallel_progress(process,inputs,64)


    


    
            
            
            

