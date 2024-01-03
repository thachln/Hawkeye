import os
import random
from pathlib import Path
import re

def transform_image_path(img_path):
    # Replace all sample '/' to '\'
    standardized_path = img_path.replace('/', '\\')
    
    # Search all charater number in string and remove space it 
    standardized_path = re.sub(r'\s*\(\s*(\d+)\s*\)\s*', r'\1', standardized_path)
    
    return standardized_path

def rename_images_in_folder(root_folder):
    for subdir, dirs, files in os.walk(root_folder):
        for filename in files:
            old_file_path = os.path.join(subdir, filename)
            new_filename = transform_image_path(filename)
            new_file_path = os.path.join(subdir, new_filename)

            if new_file_path != old_file_path:
                # Make sure there are no files with the same name
                if not os.path.exists(new_file_path):
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed "{old_file_path}" to "{new_file_path}"')
                else:
                    print(f'File "{new_file_path}" already exists. Cannot rename "{old_file_path}".')

def create_folder(folder_path):
    # Check if the directory exists or not
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the directory
        os.makedirs(folder_path)
        print(f'Folders "{folder_path}" do created.')
    else:
        print(f'Folders "{folder_path}" do exits.')

class DatasetSplitter:
    def __init__(self, dataset_dir, train_ratio=0.7):
        self.dataset_dir = Path(dataset_dir)
        self.train_ratio = train_ratio
        self.train_file = 'train.txt'
        self.val_file = 'val.txt'
        self.image_paths = []
        self.labels = {}

    def scan_dataset(self):
        for i, label_dir in enumerate(sorted(os.listdir(self.dataset_dir))):
            label_path = self.dataset_dir / label_dir
            if label_path.is_dir():
                for image_file in sorted(os.listdir(label_path)):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((i, f"{label_dir}/{image_file}"))
                self.labels[label_dir] = str(i)

    def split_dataset(self):
        random.shuffle(self.image_paths)
        num_train = int(len(self.image_paths) * self.train_ratio)
        train_images = self.image_paths[:num_train]
        val_images = self.image_paths[num_train:]
        self.write_to_file("metadata\dataset_mosquito\\"+self.train_file, train_images)
        self.write_to_file("metadata\dataset_mosquito\\"+self.val_file, val_images)

    def write_to_file(self, file_name, images):
        with open(file_name, 'w') as file:
            for label_idx, image_path in sorted(images):
                image_path =  transform_image_path(image_path)
                label = str(label_idx-1)
                line = f'{label} {image_path}\n'
                file.write(line)

    def run(self):
        self.scan_dataset()
        self.split_dataset()

if __name__ == '__main__':
    create_folder('metadata\dataset_mosquito')
    dataset_dir = 'data\dataset_mosquito'  # Replace with your actual path
    rename_images_in_folder(dataset_dir)
    splitter = DatasetSplitter(dataset_dir)
    splitter.run()
    print('Train and validation files have been created.')