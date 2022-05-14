import argparse
from tqdm import tqdm
import fiftyone as fo
import fiftyone.utils.annotations as foua


COCO_ID_LIST = [
    1,9,11,14,38,39,49,57,95,
    110,125,126,133,149,199,
    276,
    323,326,337,347,386,390,396,397,
    401,407,411,413,415,420,433,474,477,490,
    510,519,552,565,
    612,618,623
]
AREA_THRES = 0.1

def main(arg):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types = ["detections", "segmentations"],
        data_path=arg.data_dir,
        labels_path=arg.anno_dir,
        include_id=True
    )

    for sample in tqdm(dataset.take(624)):
        if sample.coco_id in COCO_ID_LIST:
            sample.tags.append("important")
        if ("FROM" in arg.anno_dir) and ("TO" in arg.anno_dir):
            bbox_area = 0.0
            for bbox in sample.detections.detections:
                bbox_area += bbox.bounding_box[2] * bbox.bounding_box[3]
                if bbox_area > AREA_THRES:
                    sample.tags.append("big_change")
                    break
        sample.save()
    
    foua.DrawConfig({"per_object_label_colors": True})
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='/opt/ml/input/data',
                        help='imageData directory')
    parser.add_argument('--anno_dir', '-a', type=str, default='/opt/ml/input/data/train_all.json', #'/opt/ml/input/code/submission/inference.json'
                        help='annotation Data directory')
    parser.add_argument('--port', '-p', type=int, default=30001,
                        help='Port Number')
    args = parser.parse_args()
    main(args)