import copy
import json
import os
import h5py
import numpy
import numpy as np
import tqdm


vg_classes = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket',
               'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book',
               'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet',
               'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
               'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye',
               'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe',
               'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet',
               'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop',
               'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain',
               'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people',
               'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player',
               'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen',
               'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink',
               'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street',
               'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower',
               'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable',
               'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

def vg_to_coco(vg_annotation_path, vg_h5_path):
    # load h5py
    vg_h5 = h5py.File(vg_h5_path, 'r')
    split = vg_h5['split'][:]  # shape: (n_images, )
    train_cutoff = numpy.where(split == 2)[0][0]
    labels = vg_h5['labels'][:]
    boxes = vg_h5['boxes_1024'][:]
    img_to_first_box = vg_h5['img_to_first_box'][:]
    numpy.append(img_to_first_box, len(labels))


    # Load Visual Genome annotations
    with open(vg_annotation_path, 'r') as f:
        image_data = json.load(f)

    # Create category dictionary to map names to ids
    category_dict = [dict(id=i, name=vg_classes[i]) for i in range(1, len(vg_classes))]

    # Initialize COCO format data
    coco_train_data = {
        "images": [],
        "annotations": [],
        "categories": category_dict
    }
    coco_test_data = copy.deepcopy(coco_train_data)
    class_out_range, class_not_provided = 0, 0

    # Convert Visual Genome to COCO format
    for index, item in enumerate(tqdm.tqdm(image_data)):
        image_id = item['image_id']
        image_data = {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "width": 800,
            "height": 600
        }
        if image_id < train_cutoff:
            coco_train_data['images'].append(image_data)
        else:
            coco_test_data['images'].append(image_data)

        for ann_index in range(img_to_first_box[index], img_to_first_box[index+1]):
            category_id = labels[ann_index, 0]
            box_1024 = boxes[ann_index]
            # scale 1024 => 800
            box = [int(point / 1024 * 800) for point in box_1024]
            # xyxy -> xywh
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            annotation = {
                "id": ann_index,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": box
            }

            if image_id < train_cutoff:
                coco_train_data['annotations'].append(annotation)
            else:
                coco_test_data['annotations'].append(annotation)

    print(f"Class out of range: {class_out_range}, "
          f"Class not provided: {class_not_provided}, "
          f"total: {len(coco_train_data['annotations']) + len(coco_test_data['annotations'])}")

    # Save COCO annotations
    vg_annotation_folder = os.path.dirname(vg_annotation_path)
    coco_save_path = os.path.join(vg_annotation_folder, 'coco_train.json')
    with open(coco_save_path, 'w') as f:
        json.dump(coco_train_data, f)
    coco_test_save_path = os.path.join(vg_annotation_folder, 'coco_test.json')
    with open(coco_test_save_path, 'w') as f:
        json.dump(coco_test_data, f)


if __name__ == '__main__':
    vg_to_coco("/home/data/datasets/vg/image_data.json",
               "/home/data/datasets/vg/VG-SGG-with-attri.h5")