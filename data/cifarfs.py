import argparse
"""
This script processes CIFAR-FS dataset by splitting it into training, validation, and test sets,
and then copying the respective data into corresponding directories.

Functions:
    get_name(root, mode_folder=True):
        Retrieves the names of directories or files in the given root directory.
        
    make_csv(data, name):
        Creates a CSV file with the given data and name.
        
    move(cls, phase):
        Copies the class data to the specified phase directory.
        
    read_csv(name):
        Reads a CSV file and returns a list of non-empty lines.
        
    get_split(cls, phase):
        Processes the class data and moves it to the specified phase directory.

Arguments:
    --data: str
        Path to the data directory. Default is "/home/wbw/PAN/cifar100".
        
    --split: str
        Path to the split folder. Default is "/home/wbw/PAN/".

Usage:
    Run the script with the appropriate arguments to process the CIFAR-FS dataset.
"""
import os
import csv
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the data', default="/home/wbw/PAN/cifar100")
parser.add_argument('--split', type=str, help='path to the split folder', default="/home/wbw/PAN/")
args = parser.parse_args()


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["filename", "label"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()


def move(cls, phase):
    shutil.copytree(os.path.join(data_root, "data", cls), os.path.join(save_root, "cifarfs", phase, cls))


def read_csv(name):
    with open(name, 'r') as f:
        split = [x.strip() for x in f.readlines() if x.strip() != '']
    return split


def get_split(cls, phase):
    record = []
    for cl in cls:
        name_imgs = get_name(os.path.join(data_root, "data", cl), mode_folder=False)
        for name in name_imgs:
            record.append([cl+"/"+name, cl])
        move(cl, phase)
    # make_csv(record, "data_split/cifar100/"+phase)


if __name__ == '__main__':
    data_root = args.data
    save_root = args.split
    r_root = os.path.join(save_root, "cifarfs")
    os.makedirs(r_root, exist_ok=True)
    name_train = os.path.join(data_root, "splits", "bertinetto", "train.txt")
    name_val = os.path.join(data_root, "splits", "bertinetto", "val.txt")
    name_test = os.path.join(data_root, "splits", "bertinetto", "test.txt")
    train = read_csv(name_train)
    val = read_csv(name_val)
    test = read_csv(name_test)
    os.makedirs(os.path.join(save_root, "cifarfs", "train"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "cifarfs", "val"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "cifarfs", "test"), exist_ok=True)
    get_split(train, "train")
    get_split(val, "val")
    get_split(test, "test")
