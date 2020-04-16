import os

import torch
from PIL import Image

from datasets.base_dataset import BaseDataset
from utils.transforms import SunSceneTransform


def parse_file(dataset_adr, categories):
    dataset = []
    with open(dataset_adr) as f:
        for line in f:
            line = line[:-1].split("/")
            category = "/".join(line[2:-1])
            file_name = "/".join(line[2:])
            if not category in categories:
                continue
            dataset.append([file_name, category])
    return dataset


def get_class_names(path):
    classes = []
    with open(path) as f:
        for line in f:
            categ = "/".join(line[:-1].split("/")[2:])
            classes.append(categ)
    class_dic = {classes[i]: i for i in range(len(classes))}
    return class_dic


class SunSceneDataset(BaseDataset):
    SUN_SCENE_INDEX_TO_NAME = None
    CLASS_WEIGHTS = None

    def __init__(self, args, data_subset):
        super(SunSceneDataset, self).__init__(args, data_subset)
        self.root_dir = args.data_path
        root_dir = self.root_dir
        train = data_subset == "train"
        if train:
            self.data_set_list = os.path.join(root_dir, "train_test", "Training_01.txt")
        else:
            self.data_set_list = os.path.join(root_dir, "train_test", "Testing_01.txt")
        categ_dict = get_class_names(os.path.join(root_dir, "ClassName.txt"))
        self.categ_dict_inverse = {val: key for key, val in categ_dict.items()}

        SunSceneDataset.SUN_SCENE_INDEX_TO_NAME = {
            scene_name: class_index for (class_index, scene_name) in categ_dict.items()
        }
        self.data_set_list = parse_file(self.data_set_list, categ_dict)
        self.data_set_list = list(
            [(path[len(category) + 1:], categ_dict[category]) for path, category in self.data_set_list]
        )
        self.transform = SunSceneTransform(self.size, data_subset)

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, "rb") as fp:
            image = Image.open(fp).convert("RGB")
        return self.transform(image)

    def get_path_and_cls(self, idx):
        file_name, category = self.data_set_list[idx]
        categ_str = self.categ_dict_inverse[category]
        path = os.path.join(self.root_dir, "all_data", categ_str, file_name)
        label = torch.tensor(category, dtype=torch.int64)
        return path, label

    def __getitem__(self, idx):
        file_name, label = self.get_path_and_cls(idx)
        image = self.load_and_resize(file_name)
        return {"data": image, "classifier_labels": label}
