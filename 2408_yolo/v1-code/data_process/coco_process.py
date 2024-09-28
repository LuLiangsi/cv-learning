import os
import json
import shutil
import argparse

if __name__ == "__main__":
    raw_path = './2408_yolo/data/coco2017_raw'
    target_path = './2408_yolo/data/coco2017'
    parser = argparse.ArgumentParser(description="create coco label")
    parser.add_argument('--json_file', type=str, default=raw_path+'/annotations/instances_train2017.json')
    parser.add_argument('--class_file', type=str, default=target_path+'/class.txt')
    parser.add_argument('--imgs_path', type=str, default=raw_path+'/train2017')
    parser.add_argument('--target_labels_path', type=str, default='./2408_yolo/data/coco2017/Train/Labels') 
    args = parser.parse_args()

    with open(args.json_file, 'r') as json_content:
        json_dict = json.load(json_content)
        images_dict_list = json_dict['images']

        labels_dict = {}

        for image_dict in images_dict_list:
            file_name = image_dict['file_name']
            image_id = image_dict['id']
            labels_dict[image_id] = {'file_name': file_name, 'bbox': []}

        annotations_dict_list = json_dict['annotations']
        for annotation_dict in annotations_dict_list:
            image_id = annotation_dict['image_id']
            bbox = annotation_dict['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            category_id = annotation_dict['category_id']
            bbox.append(category_id)
            labels_dict[image_id]['bbox'].append(bbox)

        category_index = 0
        category_id_index_map = {}

        with open(args.class_file, 'w') as category_file:
            category_dict_list = json_dict['categories']
            for category_dict in category_dict_list:
                supercategory = category_dict['supercategory']
                category_id = category_dict['id']
                category_name = category_dict['name']
                category_id_index_map[category_id] = category_index
                category_file.write(str(category_index) + ' ' + category_name + ' ' + supercategory + '\n')
                category_index += 1

        for image_id in labels_dict.keys():
            image_dict = labels_dict[image_id]
            file_name = image_dict['file_name']
            bboxes = image_dict['bbox']
            file_path = os.path.join(args.target_labels_path, file_name.replace('.jpg', '.txt'))
            with open(file_path, 'w') as file:
                for bbox in bboxes:
                    file.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(category_id_index_map[bbox[4]]) + "\n")
            
            source_img_path = os.path.join(args.imgs_path, file_name)
            target_img_path = os.path.join(args.target_labels_path.replace("/Labels", "/Imgs"))
            shutil.move(source_img_path, target_img_path)


