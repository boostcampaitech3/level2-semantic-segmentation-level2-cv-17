import cv2
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pycocotools._mask as _mask

class_labels = {
    # 0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}


def sqz(arr): # 차원 squeeze
    return np.squeeze(np.array(arr).reshape(1, -1), axis=0).tolist()


def main(arg):
    sub_df = pd.read_csv(arg.sub_csv) # 제출용 csv
    
    with open("/opt/ml/input/data/test.json", "r") as js:
        test_js = json.load(js)
    test_js['annotations'] = [] # 초기화
    
    print('Start conversion.')
    annot_id = 0
    for image_id in tqdm(range(len(sub_df))):
        mat = np.array(sub_df['PredictionString'][image_id].split(), dtype=np.uint16).reshape(256, 256) # 1x65536 to 256x256
        img = Image.fromarray(np.uint16(mat)) # array to segmentation image
        img = img.resize((512, 512)) # resize 512x512
        img = np.array(img, dtype=np.uint8) # segmentation image to array
        
        for category_id, category_name in class_labels.items():
            new_img = np.where(img!=category_id, 0, img) # 특정 class에 속하지 않으면 모두 0
            ret, binary = cv2.threshold(new_img, 0.5, 255, cv2.THRESH_BINARY) # 배경(0), 특정 claas(1)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 외곽선
            seg = list(map(sqz, contours)) # 차원, list
            R = _mask.frPoly(seg, 512, 512)
            area = _mask.area(R) # area
            bbox = _mask.toBbox(R) # bbox

            for seg_id in range(len(seg)):
                test_js['annotations'].append({
                    "id": annot_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [seg[seg_id]],
                    "area": int(area[seg_id]),
                    "bbox": bbox[seg_id].tolist(),
                    "iscrowd": 0
                })
                annot_id += 1
    print('End conversion.')
    
    print('saving a json file...')
    with open(arg.dst_json, "w") as json_file:
        json.dump(test_js, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_csv', '-s', type=str,
                        default='/opt/ml/input/level2-semantic-segmentation-level2-cv-17/torch/work_dirs/exp_19/best_miou_epoch55.csv',
                        help='submission csv path')
    parser.add_argument('--dst_json', '-d', type=str,
                        default='/opt/ml/input/level2-semantic-segmentation-level2-cv-17/torch/work_dirs/exp_19/best_miou_epoch55.json',
                        help='inference json path')
    args = parser.parse_args()
    main(args)