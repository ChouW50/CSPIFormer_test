import os, glob, random, csv, sys
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

def CreateNF(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# ImageLabel = 1 Segmentation
class ImageLbDataset(Dataset):
    def __init__(self, data_dir, split = 'train', transform = None, img_size_w = 512, img_size_h = 512, rec = False, **kwargs):
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.data_dir = data_dir
        self.split = split
        self.img_files = []
        self.label_files = []
        self.class_dict = {0: [0, 0, 0], 1: [
            255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
        
        self.transform = transform
        self.rec = rec

        with open(os.path.join(self.data_dir, f'ImageSets/Segmentation/{self.split}.txt'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            img_file = line.strip()
            label_file = os.path.splitext(img_file)[0] + '.png'
            self.img_files.append(os.path.join(
                self.data_dir, 'JPEGImages', img_file + '.jpg'))
            self.label_files.append(os.path.join(
                self.data_dir, 'SegmentationClassPNG', label_file))
        
        # self.Image = [np.array(Image.open(img_file).convert('RGB').resize(
        #     (self.img_size_w, self.img_size_h), Image.BILINEAR)) for img_file in self.img_files]
        # self.Label = [np.array(Image.open(label_file).convert('RGB').resize(
        #     (self.img_size_w, self.img_size_h), Image.BILINEAR)) for label_file in self.label_files]
        # print(f"self.Label[0].shape: {self.Label[0].shape}")
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # image_file = self.Image[idx]
        img_file = self.img_files[idx]
        # use imgaug-augmenters (iaa)
        if self.rec:
            # img_1 = np.array(img_file)
            # img_2 = np.array(img_file)
            img_1 = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            img_2 = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            if self.transform is not None:
                img_aug = self.transform(img_1)
                # img_aug = TF.resize(img_aug, (self.img_size_h // 4, self.img_size_w // 4))
                # print(img_aug.shape)
                img_2 = to_tensor(img_2)
                # print(img_2.shape)
            

            return img_aug, img_2
        # use albumentations (A)
        else:
            # label_file = self.Label[idx]
            label_file = self.label_files[idx]
            # img = np.array(img_file)
            # label = np.array(label_file)
            img = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            label = np.array(Image.open(label_file).resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            # print(f"img.shape: {img.shape}, label.shape: {label.shape}")
            # label = np.array(Image.open(label_file).resize((192, 192)))
            # label = self.label_to_color(label)
            if self.transform is not None:
                aug = self.transform(image = img, mask = label)
                img = aug["image"]
                label = aug["mask"]

            # img = torch.from_numpy(img.numpy()).float()
            # label = torch.from_numpy(label).long()

            return img, label

    def label_to_color(self, label):
        color_label = np.zeros(
            (label.shape[0], label.shape[1], 3), dtype = np.uint8)
        for i in np.unique(label):
            color_label[label == i] = self.class_dict[i]

        return color_label

    def color_to_label(self, color_label):
        label = np.zeros((color_label.shape[0], color_label.shape[1]))
        for i, rgb in self.class_dict.items():
            label[np.all(color_label == rgb, axis = 2)] = i
        return label
# ImageLabel = 3 Segmentation
class ImageLabelDataset(Dataset):
    def __init__(self, data_dir, split = 'train', transform = None, img_size_w = 512, img_size_h = 512, rec = False, **kwargs):
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.data_dir = data_dir
        self.split = split
        self.img_files = []
        self.label_files = []
        self.class_dict = {0: [0, 0, 0], 1: [
            255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
        
        self.transform = transform
        self.rec = rec
        total_img_files = os.listdir(f"{data_dir}{split}_seg\\")
        # totl_label_files = os.listdir(f"{data_dir}{split}_mask\\")
        for i in range(len(total_img_files)):
            self.img_files.append(os.path.join(f"{data_dir}{split}_seg\\",total_img_files[i]))
            self.label_files.append(os.path.join(f"{data_dir}{split}_mask\\",total_img_files[i]))
        # used for albumentations (A)
        # self.Image = [np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR)) for img_file in self.img_files]
        # self.Label = [np.array(Image.open(label_file).resize((self.img_size_w, self.img_size_h), Image.BILINEAR)) for label_file in self.label_files]
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # image_file = self.Image[idx]
        img_file = self.img_files[idx]
        # use imgaug-augmenters (iaa)
        if self.rec:
            # img_1 = np.array(img_file)
            # img_2 = np.array(img_file)
            img_1 = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            img_2 = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            if self.transform is not None:
                img_aug = self.transform(img_1)
                # img_aug = TF.resize(img_aug, (self.img_size_h // 4, self.img_size_w // 4))
                # print(img_aug.shape)
                img_2 = to_tensor(img_2)
                # print(img_2.shape)
            

            return img_aug, img_2
        # use albumentations (A)
        else:
            # label_file = self.Label[idx]
            label_file = self.label_files[idx]
            # img = np.array(img_file)
            # label = np.array(label_file)
            img = np.array(Image.open(img_file).convert('RGB').resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            label = np.array(Image.open(label_file).resize((self.img_size_w, self.img_size_h), Image.BILINEAR))
            # print(f"img.shape: {img.shape}, label.shape: {label.shape}")
            # label = np.array(Image.open(label_file).resize((192, 192)))
            # label = self.label_to_color(label)
            if self.transform is not None:
                aug = self.transform(image = img, mask = label)
                img = aug["image"]
                label = aug["mask"]
            # img = torch.from_numpy(img.numpy()).float()
            # label = torch.from_numpy(label).long()

            return img, label
        # # Only save the first channel
        # # label = np.array(label)[:, :, 0]
        # # For grayscale (because of the label file is RGBA image)
        # # label = 0.2989 * label[:, :, 0] + 0.5870 * label[:, :, 1] + 0.1140 * label[:, :, 2]
        # # print(f"img.shape: {img.shape}, label.shape: {label.shape}")
        # # label = np.array(Image.open(label_file).resize((192, 192)))
        # # label = self.label_to_color(label)
        # if self.transform is not None:
        #     # used for albumentations (A)
        #     aug = self.transform(image = self.Image[idx], mask = self.Label[idx])
        #     img = aug["image"]
        #     label = aug["mask"]
        #     # end =====
        #     # used for albumentations (A)
        #     return img, label

    def label_to_color(self, label):
        color_label = np.zeros(
            (label.shape[0], label.shape[1], 3), dtype = np.uint8)
        for i in np.unique(label):
            color_label[label == i] = self.class_dict[i]

        return color_label

    def color_to_label(self, color_label):
        label = np.zeros((color_label.shape[0], color_label.shape[1]))
        for i, rgb in self.class_dict.items():
            label[np.all(color_label == rgb, axis = 2)] = i
        return label
# ImageLabel = 0 Classification
class TxtLabelDataset(Dataset):
    def __init__(self, data_dir, size, split = 'train', transform = None, classnum = None, ratio_train = None, ratio_val = None, ratio_train_ = None):
        super().__init__()
        self.size = size
        if ratio_train is None or ratio_val is None:
            self.data_dir = data_dir + f"{split}\\"
            data_csv = split
            split = ''
        else:
            self.data_dir = data_dir + f"{split}\\"
            data_csv = split
        self.transform = transform
        self.name2label = {}
        file_name = data_dir.split(os.sep)[-2]
        all_classes = sorted(os.listdir(self.data_dir))
        for name in all_classes[:classnum]:
            if not os.path.isdir(os.path.join(self.data_dir, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        CreateNF(f"{data_dir}cls_{classnum}\\")
        self.img_files, self.label_files = self.load_csv(f"{data_dir}cls_{classnum}\\{file_name}_{data_csv}.csv")
        print(f"{int(ratio_train_ * len(self.img_files))}to{int(ratio_train * len(self.img_files))}")
        if split == 'train':
            self.img_files = self.img_files[int(ratio_train_ * len(self.img_files)):int(ratio_train * len(self.img_files))]
            self.label_files = self.label_files[int(ratio_train_ * len(self.img_files)):int(ratio_train * len(self.label_files))]
        elif split == 'val':
            self.img_files = self.img_files[int(ratio_train * len(self.img_files)):int((ratio_train + ratio_val) * len(self.img_files))]
            self.label_files = self.label_files[int(ratio_train * len(self.label_files)):int((ratio_train + ratio_val) * len(self.label_files))]
        elif split == 'test':
            self.img_files = self.img_files[int((ratio_train + ratio_val) * len(self.img_files)):]
            self.label_files = self.label_files[int((ratio_train + ratio_val) * len(self.label_files)):]
        else:
            self.img_files = self.img_files[:]
            self.label_files = self.label_files[:]
        self.Image = [np.array(Image.open(img_file).convert('RGB').resize(
            (self.size[1], self.size[0]), Image.BILINEAR)) for img_file in self.img_files]
        print(f"path: {self.img_files[0]}, {self.img_files[len(self.img_files) - 1]}")
    def load_csv(self, filename):
        if not os.path.exists(filename):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.data_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.data_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.data_dir, name, '*.jpeg'))
            random.shuffle(images)
            with open(filename, mode = 'w', newline = '') as f:
                write = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    write.writerow([img, label])
                print(f"Finish the image & label path to {filename}")
        images, labels = [], []
        with open(filename) as f:
            read = csv.reader(f)
            for r in read:
                img, label = r
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        # print(f"img_files: {self.img_files[idx]}")
        img, label = self.img_files[idx], self.label_files[idx]
        # For albumentations
        # img, label = self.Image[idx], self.label_files[idx]
        # =====end=====
        # -----start-----
        if self.transform is not None:
            # For albumentations
            # aug = self.transform(image = img)
            # img = aug["image"]
            # =====end=====
            # for transforms.Compose
            img = self.transform(img)
            # =====end=====
        label = torch.tensor(label)
        # -----end-----
        return img, label
# ImageLabel = 2 NoLabel
class NoLabelDataset (Dataset):
    def __init__(self, data_dir, size, split = 'train', transform = None, classnum = None, ratio_train = None, ratio_val = None, ):
        super().__init__()
        self.size = size
        if ratio_train or ratio_val is None:
            self.data_dir = data_dir + f"{split}\\"
            data_csv = split
            split = ''
        else:
            self.data_dir = data_dir
            data_csv = split
        self.transform = transform
        file_name = data_dir.split(os.sep)[-2]
        CreateNF(f"{data_dir}cls_{classnum}\\")
        self.img_files = self.load_csv(f"{data_dir}test_results.csv")
        self.Image = [np.array(Image.open(img_file).convert('RGB').resize(
            (self.size[1], self.size[0]), Image.BILINEAR)) for img_file in self.img_files]
    def sortedkey(self, filename):
        parts = filename.split('_')
        prefix = parts[0]
        number = int(parts[1].split('.')[0])
        return (prefix, number)
    def load_csv(self, filename):
        if not os.path.exists(filename):
            all_names = os.listdir(self.data_dir)
            all_names.sort(key = self.sortedkey)
            with open(filename, mode = 'w', newline = '') as f:
                write = csv.writer(f)
                write.writerow(["id", "label"])
                for name in all_names:
                    nm, num = name.split('_')[0], name.split('_')[1]
                    if num[:-4] == '0':
                        write.writerow([nm, 0])
                    write.writerow([f"{nm}_{num[:-4]}", 0])
                print(f"Finish the image & label path to {filename}")
        images = []
        with open(filename) as f:
            read = csv.reader(f)
            next(read)
            for r in read:
                img, id = r
                if '_' in img:
                    images.append(f"{self.data_dir}{img}.png")
        # print(images)
        return images
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img = self.img_files[idx]
        # For albumentations
        # img = self.Image[idx]
        # =====end=====
        # -----start-----
        if self.transform is not None:
            # For albumentations
            # aug = self.transform(image = img)
            # img = aug["image"]
            # =====end=====
            # for transforms.Compose
            img = self.transform(img)
            # =====end=====
        # -----end-----
        return img
