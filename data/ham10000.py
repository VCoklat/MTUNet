import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

def make_csv(data, name):
    data.to_csv(name + ".csv", index=False)

def create_data_splits(data_dir, split_dir):
    metadata = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    train, test = train_test_split(metadata, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    
    for df, split in zip([train, val, test], ['train', 'val', 'test']):
        make_csv(df, os.path.join(split_dir, split))

def move_images(data_dir, split_dir):
    for split in ['train', 'val', 'test']:
        split_df = pd.read_csv(os.path.join(split_dir, split + ".csv"))
        for _, row in split_df.iterrows():
            img_name = row['image_id'] + '.jpg'
            src_path = os.path.join(data_dir, img_name)
            dst_dir = os.path.join(split_dir, split, row['dx'])
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, dst_dir)

if __name__ == '__main__':
    data_root = 'FSL_Data/ham10000'
    split_root = 'FSL_Data/ham10000'
    create_data_splits(data_root, split_root)
    move_images(data_root, split_root)
