import sys
import skimage
from numpy import ndarray
from skimage import io, transform, util
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# Define the class for image augmentation
class ImageAugmenter:
    def __init__(self, folder_path: str, augmented_path: str, target_images_per_folder: int = 128):
        self.folder_path = Path(folder_path)
        self.augmented_path = Path(augmented_path)
        self.target_images_per_folder = target_images_per_folder

        self.available_transformations = {
            'rotate': self.random_rotation,
            'noise': self.random_noise,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip
        }

    def augment_images(self):
        self.augmented_path.mkdir(parents=True, exist_ok=True)

        for folder in self.folder_path.iterdir():
            if not folder.is_dir():
                continue

            augmented_subfolder_path = self.augmented_path / folder.name
            augmented_subfolder_path.mkdir(parents=True, exist_ok=True)

            num_augmented_images = len(list(augmented_subfolder_path.glob('*.jpg')))  # Assuming .jpg images
            new_images_needed = self.target_images_per_folder - num_augmented_images

            if new_images_needed > 0:
                original_images = list(folder.glob('*.jpg'))
                if not original_images:
                    print(f"No images found in {folder}. Skipping...")
                    continue

                while num_augmented_images < self.target_images_per_folder:
                    for file in original_images:
                        for transform_key in self.available_transformations:
                            if num_augmented_images >= self.target_images_per_folder:
                                break

                            img = io.imread(file)
                            transformed_image = self.available_transformations[transform_key](img)
                            transformed_image = transformed_image.astype(float)
                            transformed_image /= np.max(transformed_image)
                            transformed_image = (transformed_image * 255).astype(np.uint8)

                            new_file_name = f"{transform_key}_{file.stem}_{num_augmented_images}.jpg"

                            new_file_path = augmented_subfolder_path / new_file_name
                            io.imsave(new_file_path, transformed_image)

                            num_augmented_images += 1

    # Transformation functions
    def random_rotation(self, image_array: np.ndarray) -> np.ndarray:
        random_degree = random.uniform(-25, 25)
        return transform.rotate(image_array, random_degree)

    def random_noise(self, image_array: np.ndarray) -> np.ndarray:
        return util.random_noise(image_array)  # Assuming util.random_noise is defined elsewhere

    def horizontal_flip(self, image_array: np.ndarray) -> np.ndarray:
        return np.fliplr(image_array)

    def vertical_flip(self, image_array: np.ndarray) -> np.ndarray:
        return np.flipud(image_array)

if __name__ == '__main__':
    # Example usage
    #  data_augmentation.py <folder_path> <augmented_path>
    if len(sys.argv) < 2:
        print('Usage: data_augmentation.py [folder_path] [augmented_path]')
        print('Default: folder_path=./data/dataset/; ./data/dataset_mosquito/')

    folder_path = sys.argv[1] if len(sys.argv) > 1 else './data/dataset/'
    augmented_path = sys.argv[2] if len(sys.argv) > 2 else './data/dataset_mosquito/'

    print('Running with folder_path=%s, augmented_path=%s\n' % (folder_path, augmented_path))

    augmenter = ImageAugmenter(folder_path=folder_path, augmented_path=augmented_path)
    augmenter.augment_images()
