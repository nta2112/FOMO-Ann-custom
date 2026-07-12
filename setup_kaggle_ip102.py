# ------------------------------------------------------------------------
# IP102 Dataset Setup for FOMO OWOD Continual Learning on Kaggle
# ------------------------------------------------------------------------

import os
import json
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('IP102 Dataset Setup', add_help=False)
    parser.add_argument('--image_dir', default='/kaggle/input/datasets/rtlmhjbn/ip02-dataset/classification', type=str,
                        help='Path to the Kaggle image directory')
    parser.add_argument('--ann_dir', default='/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations', type=str,
                        help='Path to the Kaggle COCO annotations directory')
    parser.add_argument('--output_root', default='./data/OWOD', type=str,
                        help='Path to destination data folder (under data/OWOD)')
    return parser

def main(args):
    print("Starting dataset setup...")
    print(f"Image directory: {args.image_dir}")
    print(f"Annotation directory: {args.ann_dir}")
    print(f"Output root: {args.output_root}")

    # 1. Create directories
    img_dest_dir = os.path.join(args.output_root, 'JPEGImages', 'IP102')
    ann_dest_dir = os.path.join(args.output_root, 'Annotations', 'IP102')
    set_dest_dir = os.path.join(args.output_root, 'ImageSets', 'IP102')

    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(ann_dest_dir, exist_ok=True)
    os.makedirs(set_dest_dir, exist_ok=True)

    # 2. Walk the image directory to build a fast map of filename -> absolute path
    print("Scanning input images...")
    image_path_map = {}
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Store the direct path. Filenames are unique in IP102.
                image_path_map[file] = os.path.join(root, file)

    print(f"Found {len(image_path_map)} source images in total.")

    # 3. Process COCO JSON files and generate XML annotations
    splits = ['train', 'val', 'test']
    all_categories = set()
    category_id_to_name = {}

    # Read train JSON to extract and build task classes (since it contains all categories)
    train_json_path = os.path.join(args.ann_dir, 'train.json')
    if not os.path.exists(train_json_path):
        print(f"Error: {train_json_path} does not exist.")
        return

    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
        for cat in train_data['categories']:
            category_id_to_name[cat['id']] = cat['name']
            all_categories.add(cat['name'])

    # Sort categories alphabetically to ensure consistent class lists
    sorted_classes = sorted(list(all_categories))
    num_classes = len(sorted_classes)
    print(f"Total categories identified: {num_classes}")
    assert num_classes == 102, f"Expected 102 categories, found {num_classes} instead."

    # 4. Generate OWOD task split files
    # Task 1: 27 classes, Tasks 2-4: 25 classes each
    t1_classes = sorted_classes[:27]
    t2_classes = sorted_classes[27:52]
    t3_classes = sorted_classes[52:77]
    t4_classes = sorted_classes[77:102]

    # Helper function to write class lists
    def write_class_list(filename, classes):
        with open(os.path.join(set_dest_dir, filename), 'w') as f:
            for cls in classes:
                f.write(cls + '\n')
        print(f"Wrote {filename} with {len(classes)} classes.")

    write_class_list('t1_known.txt', t1_classes)
    write_class_list('t2_known.txt', t1_classes + t2_classes)
    write_class_list('t3_known.txt', t1_classes + t2_classes + t3_classes)
    write_class_list('t4_known.txt', sorted_classes)

    write_class_list('t1_unknown_classnames_groundtruth.txt', t2_classes + t3_classes + t4_classes)
    write_class_list('t2_unknown_classnames_groundtruth.txt', t3_classes + t4_classes)
    write_class_list('t3_unknown_classnames_groundtruth.txt', t4_classes)

    # Write best_templates.txt
    templates = [
        "itap of a {c}.",
        "a bad photo of the {c}.",
        "a origami {c}.",
        "a photo of the large {c}.",
        "a {c} in a video game.",
        "art of the {c}.",
        "a photo of the small {c}."
    ]
    with open(os.path.join(set_dest_dir, 'best_templates.txt'), 'w') as f:
        for t in templates:
            f.write(t + '\n')
    print("Wrote best_templates.txt")

    # 5. Process XML annotations and symlink images for each split
    for split in splits:
        json_path = os.path.join(args.ann_dir, f'{split}.json')
        if not os.path.exists(json_path):
            print(f"Warning: Annotation split {json_path} not found. Skipping.")
            continue

        print(f"Processing split: {split}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Update category mapping in case of missing categories in train.json
        for cat in data['categories']:
            category_id_to_name[cat['id']] = cat['name']

        # Group annotations by image_id
        img_anns = defaultdict(list)
        for ann in data['annotations']:
            img_anns[ann['image_id']].append(ann)

        split_images_list = []
        symlink_count = 0
        xml_count = 0

        for img in data['images']:
            filename = img['file_name']
            base_name = os.path.basename(filename)
            width = img['width']
            height = img['height']
            img_id = img['id']

            # Find source path of the image by base name
            src_path = image_path_map.get(base_name)

            if not src_path:
                continue

            # Create symlink to image (flat structure)
            dest_img_path = os.path.join(img_dest_dir, base_name)
            if not os.path.exists(dest_img_path):
                os.symlink(src_path, dest_img_path)
                symlink_count += 1

            # Generate VOC XML Annotation
            xml_filename = os.path.splitext(base_name)[0] + '.xml'
            dest_xml_path = os.path.join(ann_dest_dir, xml_filename)

            root_el = ET.Element("annotation")
            ET.SubElement(root_el, "filename").text = base_name
            size_el = ET.SubElement(root_el, "size")
            ET.SubElement(size_el, "width").text = str(width)
            ET.SubElement(size_el, "height").text = str(height)
            ET.SubElement(size_el, "depth").text = "3"

            for ann in img_anns[img_id]:
                obj_el = ET.SubElement(root_el, "object")
                ET.SubElement(obj_el, "name").text = category_id_to_name[ann['category_id']]
                ET.SubElement(obj_el, "difficult").text = "0"
                bb_el = ET.SubElement(obj_el, "bndbox")
                # COCO box format: [x, y, w, h] (0-indexed)
                # VOC format: [xmin, ymin, xmax, ymax] (1-indexed)
                bbox = ann['bbox']
                xmin = int(bbox[0] + 1.0)
                ymin = int(bbox[1] + 1.0)
                xmax = int(bbox[0] + bbox[2] + 1.0)
                ymax = int(bbox[1] + bbox[3] + 1.0)
                ET.SubElement(bb_el, "xmin").text = str(xmin)
                ET.SubElement(bb_el, "ymin").text = str(ymin)
                ET.SubElement(bb_el, "xmax").text = str(xmax)
                ET.SubElement(bb_el, "ymax").text = str(ymax)

            # Write XML file
            tree = ET.ElementTree(root_el)
            # Indent to match VOC layout
            if hasattr(ET, 'indent'):
                ET.indent(tree, space="  ")
            tree.write(dest_xml_path, encoding='utf-8', xml_declaration=False)
            xml_count += 1

            split_images_list.append(base_name)

        # Write txt list of file names
        txt_filename = f'{split}.txt'
        # If the split is val, we should also write it as test.txt if test split is missing
        # But here we write val.txt and test.txt separately.
        with open(os.path.join(set_dest_dir, txt_filename), 'w') as f:
            for fn in split_images_list:
                f.write(fn + '\n')

        print(f"Finished split: {split}. Created {symlink_count} symlinks, {xml_count} XML files.")

    print("IP102 Dataset setup successfully completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('IP102 dataset generator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
