# 해야할 거를 하나씩 적자.
# 1. 기존의 데이터들을 이름 바꿔서 저기에 넣는거
# 이미지들을 이름 바꿔서 넣고, 그거에 맞게 json도 수정해주자
# 2. 그런 다음에 이 둘을 이용해서 mask map을 만들자. 


# 1
# train.json을 읽어서 저기 있는 경로대로 이미지들을 하나씩 방문하면서 걔를 복사하고 이름은 변경해서
# 우리가 원하는 경로에 저장
# 여기서 자동으로 mkdir도 해주게 하면 되겠다

# 하는김에 바꾼 json도 넣어둘까?

import json
import os
import shutil
from pycocotools.coco import COCO
import cv2
import numpy as np




data_root = '/opt/ml/input/data'

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

category_names  = ["Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def copy_img(json_path ):
    json_name = json_path.split('/')[-1].split('.')[0]
    with open(json_path, 'r') as f :
        json_data = json.load(f)

    # 이게 어떤 json인지에 따라서 달라짐.

    for folder in ['mmseg/img_dir', 'mmseg/ann_dir']:
        os.makedirs(os.path.join(data_root,os.path.join(folder,json_name)), exist_ok=True)
    
    for image in json_data['images']:
        shutil.copyfile(os.path.join(data_root,image['file_name']),
                         os.path.join(data_root,f"mmseg/img_dir/{json_name}/{str(image['id']).zfill(4)}.jpg"))
def gen_mask(json_path):
    json_name = json_path.split('/')[-1].split('.')[0]
    coco = COCO(json_path)

    image_ids = coco.getImgIds()
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]

        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)

        # Background = 0
        masks = np.zeros((image_info["height"], image_info["width"]))
        # General trash = 1, ... , Cigarette = 10
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
        for i in range(len(anns)):
            className = get_classname(anns[i]['category_id'], cats)
            pixel_value = category_names.index(className)
            masks[coco.annToMask(anns[i]) == 1] = pixel_value
        masks = masks.astype(np.int8)

        cv2.imwrite(os.path.join(data_root,f"mmseg/ann_dir/{json_name}/{str(image_info['id']).zfill(4)}.png"), masks)





if __name__ =='__main__' :
    copy_img('/opt/ml/input/data/train.json')
    gen_mask('/opt/ml/input/data/train.json')
    copy_img('/opt/ml/input/data/val.json')
    gen_mask('/opt/ml/input/data/val.json')
    copy_img('/opt/ml/input/data/test.json')



# 결과적으로는 원래 이미지 만큼의 크기의 제로를 만든다음에 그걸
# 클래스랑 mask에 맞게 마스크이미지로 만들어주면 되는 것 같음.

