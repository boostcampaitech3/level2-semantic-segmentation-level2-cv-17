import json
import cv2
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import pycocotools._mask as _mask

def main(arg):
    sub_df = pd.read_csv(arg.sub_csv) # 제출한 csv
    
    # test.json
    with open("../data/test.json", "r") as js:
        test_js = json.load(js)
    test_js['annotations'] = [] # 초기화
    
    def sqz(arr): # 차원 squeeze
        return np.squeeze(arr.reshape(1, -1), axis=0).tolist()
    print('Start conversion.')
    n_id = 0
    for i in tqdm(range(len(sub_df))):
        mat = np.array(sub_df['PredictionString'][i].split(), dtype=np.uint16).reshape(256, 256) # 1x65536 to 256x256
        img = Image.fromarray(np.uint16(mat)*25) # array to segmentation image
        img = img.resize((512, 512)) # resize 512x512
        img = np.array(img, dtype=np.uint8) # segmentation image to array
        ret, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY) # 배경(0), 물체(1~10)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 외곽선
        seg = list(map(sqz, contours)) # 차원, list
        R = _mask.frPoly(seg, 512, 512)
        area = _mask.area(R) # area
        bbox = _mask.toBbox(R) # bbox
        for j in range(len(seg)):
            test_js['annotations'].append({
                "id": n_id,
                "image_id": i,
                "category_id": int(mat[seg[j][1]//2][seg[j][0]//2]),
                "segmentation": [seg[j]],
                "area": int(area[j]),
                "bbox": bbox[j].tolist(),
                "iscrowd": 0
            })
            n_id += 1
        time.sleep(0.01)
    print('End conversion.')
    print('saving a json file...')
    with open(arg.dst_json, "w") as json_file:
        json.dump(test_js, json_file, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_csv', '-s', type=str, default='/opt/ml/input/code/submission/fcn_resnet50_best_model(pretrained).csv',
                        help='submission csv path')
    parser.add_argument('--dst_json', '-d', type=str, default='/opt/ml/input/code/submission/inference.json',
                        help='inference json path')
    args = parser.parse_args()
    main(args)