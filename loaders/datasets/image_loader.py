import os
import pandas as pd
import PIL.Image as Image

__all__ = ['DatasetFolder']


class DatasetFolder(object):

    def __init__(self, root, set_name, split_type, transform, out_name=False, cls_selction=None):
        assert split_type in ['train', 'test', 'val']
        split_file = os.path.join("data/data_split", set_name, split_type + '.csv')
        assert os.path.isfile(split_file)
        data = self.read_csv(split_file)
        cls = list(data.keys())

        data_new, cls_new = self.select_class(data, cls, cls_selction)
        self.data, self.labels = [x[0] for x in data_new], [cls_new.index(x[1]) for x in data_new]
        self.split_type = split_type
        self.set_name = set_name
        self.root = root + "/" + set_name
        self.transform = transform
        self.out_name = out_name
        self.length = len(self.data)

    def select_class(self, data, cls, selection):
        current_data = []
        if selection is not None:
            cls_new = [cls[x] for x in selection]
        else:
            cls_new = cls
        for cl in cls_new:
            current_data.extend(data[cl])
        return current_data, cls_new

    def make_dict(self, all_data):
        cls = {}
        for img in all_data:
            if img[1] not in cls:
                cls.update({img[1]: [[img[0], img[1]]]})
            else:
                cls[img[1]].append([img[0], img[1]])
        return cls

    def read_csv(self, name):
        data_all = pd.read_csv(name)
        data = data_all[["filename", "label"]]
        data = data.values
        return self.make_dict(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.set_name == "miniImageNet":
            folder_name = self.data[index][:9]
            img_name = self.data[index]
        elif self.set_name == "CUB200":
            folder_name, img_name = self.data[index].split("/")
        img_root = os.path.join(self.root, self.split_type, folder_name, img_name)
        assert os.path.isfile(img_root)
        img = Image.open(img_root).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, img_name
        else:
            return img, label