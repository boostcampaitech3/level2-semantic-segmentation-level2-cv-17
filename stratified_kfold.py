import os
import os.path as osp
import argparse
import random
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

data_path = osp.join("..", "data")
annotations_path = osp.join(data_path, "train_all.json")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_split", "-n", type=int, default=5)
    parser.add_argument("--path", "-p", type=str, default=osp.join("..", "data", "stratified"))
    args = parser.parse_args()
    args.path += f"_{args.n_split}fold"
    return args


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)


def main(args):
    set_seed(args.seed)
    if not osp.exists(args.path):
        os.mkdir(args.path)

    with open(annotations_path, "r") as f:
        data = json.loads(f.read())
        images = data["images"]
        categories = data["categories"]
        annotations = data["annotations"]

    annotations_df = pd.DataFrame.from_dict(annotations)
    x = images
    y = [[0] * len(categories) for _ in range(len(images))]

    for _, anno in enumerate(annotations):
        image_id = anno["image_id"]
        category_id = anno["category_id"] - 1
        y[image_id][category_id] += 1

    mskf = MultilabelStratifiedKFold(n_splits=args.n_split, shuffle=True)

    for idx, (train_index, val_index) in tqdm(
        enumerate(mskf.split(x, y)), total=args.n_split
    ):
        train_dict = dict()
        val_dict = dict()

        for i in ["info", "licenses", "categories"]:
            train_dict[i] = data[i]
            val_dict[i] = data[i]

        train_dict["images"] = np.array(images)[train_index].tolist()
        val_dict["images"] = np.array(images)[val_index].tolist()

        train_dict["annotations"] = annotations_df[
            annotations_df["image_id"].isin(train_index)
        ].to_dict("records")
        val_dict["annotations"] = annotations_df[
            annotations_df["image_id"].isin(val_index)
        ].to_dict("records")

        train_dir = osp.join(args.path, f"train_fold{idx}.json")
        val_dir = osp.join(args.path, f"val_fold{idx}.json")

        with open(train_dir, "w") as train_file:
            json.dump(train_dict, train_file, indent=4)

        with open(val_dir, "w") as val_file:
            json.dump(val_dict, val_file, indent=4)

    print("Done Make files")


def update_dataset(index, mode, input_json, output_dir):

    with open(input_json, "r") as file:
        data = json.load(file)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    image_ids = [x.get("id") for x in images]
    image_ids.sort()

    new_image_ids = set(image_ids)
    new_images = [x for x in images if x.get("id") in new_image_ids]

    train_id2id = dict()

    for i in range(len(new_images)):
        train_id2id[new_images[i]["id"]] = i
        new_images[i]["id"] = i

    new_annotations = [x for x in annotations if x.get("image_id") in new_image_ids]

    for i in range(len(new_annotations)):
        new_annotations[i]["image_id"] = train_id2id[new_annotations[i]["image_id"]]

    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    output_json = osp.join(output_dir, f"{mode}_fold{index}.json")
    with open(output_json, "w") as new_file:
        json.dump(new_data, new_file, indent=4)
    
    print(f"update {output_json}")


def update_loop(args):
    for i in range(args.n_split):
        update_dataset(
            index=i,
            mode="train",
            input_json=osp.join(args.path, f"train_fold{i}.json"),
            output_dir=args.path,
        )
        update_dataset(
            index=i,
            mode="val",
            input_json=osp.join(args.path, f"val_fold{i}.json"),
            output_dir=args.path,
        )


if __name__ == "__main__":
    args = get_parser()
    main(args)
    update_loop(args)