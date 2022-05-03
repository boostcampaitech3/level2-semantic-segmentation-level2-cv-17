import fiftyone as fo
import argparse

def main(arg):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types = ["detections","segmentations"],
        data_path=arg.data_dir,
        labels_path=arg.anno_dir,
        include_id  =True
    )
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")

    session.config.color_pool = ["#7B241C", "#9B59B6 ","#AED6F1","#73C6B6","#2ECC71","#F7DC6F","#E59866","#AEB6BF","#1B2631","#D2B4DE"]
    session.refresh()   
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