import cv2
import os
import time
import random
import imagesize
import numpy as np
from utils import image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class coco_classify_dataset(Dataset):
    def __init__(self,imgs_path = "./2408_yolo/data/coco2017/Train/Imgs", txts_path = "./2408_yolo/data/coco2017/Train//Labels", is_train = True, edge_threshold=200, class_num=80, input_size=256):  # input_size:输入图像的尺度
        img_names = os.listdir(txts_path)
        self.is_train = is_train

        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

        self.input_size = input_size
        self.train_data = []  # [img_path,[[coord, class_id]]]

        for img_name in img_names:
            if img_name == ".DS_Store":
                continue
            img_path = os.path.join(imgs_path, img_name.replace(".txt", ".jpg"))
            txt_path = os.path.join(txts_path, img_name)

            coords = []

            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as label_txt:
                for label in label_txt:
                    label = label.replace("\n", "").split(" ")
                    class_id = int(label[4])

                    if class_id >= class_num:
                        continue

                    xmin = round(float(label[0]))
                    ymin = round(float(label[1]))
                    xmax = round(float(label[2]))
                    ymax = round(float(label[3]))

                    if (xmax - xmin) < edge_threshold or (ymax - ymin) < edge_threshold:
                        continue

                    coords.append([xmin, ymin, xmax, ymax, class_id])

            if len(coords) != 0:
                self.train_data.append([img_path, coords])

    def __getitem__(self, item):

        img_path, coords = self.train_data[item]
        img = cv2.imread(img_path)
        random.seed(int(time.time()))
        random_index = random.randint(0, len(coords) - 1)
        xmin, ymin, xmax, ymax, class_index = coords[random_index]
        img = img[ymin: ymax, xmin: xmax]

        if self.is_train:
            transform_seed = random.randint(0, 2)

            if transform_seed == 0:  # 原图
                img = image.resize_image_without_annotation(img, self.input_size, self.input_size)

            elif transform_seed == 1:  # 缩放+中心裁剪
                img, coords = image.center_crop_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)

            elif transform_seed == 2:  # 明度调整 YOLO在论文中称曝光度为明度
                img = image.resize_image_without_annotation(img, self.input_size, self.input_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)

        else:
            img = image.resize_image_without_annotation(img, self.input_size, self.input_size)

        img = self.transform_common(img)
        return img, class_index

    def __len__(self):
        return len(self.train_data)