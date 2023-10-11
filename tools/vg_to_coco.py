import copy
import ujson as json
import os
import h5py
import numpy
import numpy as np
import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial


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

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def vg_to_coco(vg_annotation_path, vg_h5_path):
    # load h5py
    vg_h5 = h5py.File(vg_h5_path, 'r')
    split = vg_h5['split'][:]  # shape: (n_images, )
    train_cutoff = numpy.where(split == 2)[0][0]
    labels = vg_h5['labels'][:]
    boxes = vg_h5['boxes_1024'][:]
    boxes_512 = vg_h5['boxes_512'][:]
    img_to_first_box = vg_h5['img_to_first_box'][:]
    img_to_first_box = numpy.append(img_to_first_box, [len(labels)])

    for index in range(len(img_to_first_box) - 1, -1, -1):
        if img_to_first_box[index] == -1:
            img_to_first_box[index] = img_to_first_box[index + 1]

    # Load Visual Genome annotations
    with open(vg_annotation_path, 'r') as f:
        vg_data = json.load(f)

    # Create category dictionary to map names to ids
    category_dict = [dict(id=i, name=vg_classes[i], supercategory=vg_classes[i]) for i in range(1, len(vg_classes))]

    # Initialize COCO format data
    coco_train_data = {
        "images": [],
        "annotations": [],
        "categories": category_dict
    }
    coco_test_data = copy.deepcopy(coco_train_data)

    # Convert Visual Genome to COCO format
    for index, item in enumerate(tqdm.tqdm(vg_data)):
        image_id = item['image_id']
        filename = item['url'].split('/')[-1]
        # Read image size
        with open(os.path.join("/home/data/datasets/vg/VG_100K", filename), 'rb') as f:
            with Image.open(f) as im:
                im = _apply_exif_orientation(im)
                width, height = im.size
        image_data = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        }
        if index < train_cutoff:
            coco_train_data['images'].append(image_data)
        else:
            coco_test_data['images'].append(image_data)

        max_size = max(item['width'], item['height'])
        for ann_index in range(img_to_first_box[index], img_to_first_box[index+1]):
            category_id = labels[ann_index, 0]
            box = boxes[ann_index]
            box[:2] = box[:2] - box[2:] / 2
            # box[2:] = box[:2] + box[2:]
            box = box.astype(float) / 1024 * max(height, width)
            box = box.astype(int).tolist()
            area = box[2] * box[3]
            annotation = {
                "id": ann_index,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": box,
                "iscrowd": 0,
                "area": area,
            }

            if index < train_cutoff:
                coco_train_data['annotations'].append(annotation)
            else:
                coco_test_data['annotations'].append(annotation)

    # Save COCO annotations
    vg_annotation_folder = os.path.dirname(vg_annotation_path)
    coco_save_path = os.path.join(vg_annotation_folder, 'coco_train.json')
    with open(coco_save_path, 'w') as f:
        json.dump(coco_train_data, f)
    coco_test_save_path = os.path.join(vg_annotation_folder, 'coco_test.json')
    with open(coco_test_save_path, 'w') as f:
        json.dump(coco_test_data, f)

def fix_resolution(image_data_path, coco_path):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    with open(image_data_path, 'r') as f:
        image_data = json.load(f)

    mapping = {}
    for image in image_data:
        filename = image['url'].split('/')[-1]
        mapping[filename] = image

    vg_path = "/home/data/datasets/vg/VG_100K"
    for image in tqdm.tqdm(coco_data['images']):
        filename = image['file_name']
        try:
            image['width'] = mapping[filename]['width']
            image['height'] = mapping[filename]['height']
        except KeyError:
            with Image.open(os.path.join(vg_path, filename)) as im:
                image['width'] = im.width
                image['height'] = im.height


    with open(coco_path, 'w') as f:
        json.dump(coco_data, f)


if __name__ == '__main__':
    # fix_resolution("/home/data/datasets/vg/image_data.json",
    #                "/home/data/datasets/vg/coco_train.json")
    vg_to_coco("/home/data/datasets/vg/image_data.json",
               "/home/data/datasets/vg/VG-SGG-with-attri.h5")