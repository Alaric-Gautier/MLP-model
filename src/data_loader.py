import os
import cv2
import numpy as np
from skimage.feature import hog
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PneumoniaDataset(Dataset):
    def __init__(self, path_to_images, categories, img_size=(128, 128)):
        self.path_to_images = path_to_images
        self.categories = categories
        self.img_size = img_size
        self.data = []
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(self.img_size),
            # Ajoutez d'autres transformations si nécessaire
        ])
        self.load_data()

    def extract_hog_features(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_image, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), feature_vector=True)
        return hog_features

    def load_data(self):
        for category in self.categories:
            path = os.path.join(self.path_to_images, category)
            class_num = self.categories.index(category)

            for img in os.listdir(path):
                try:
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(path, img)
                        img_array = cv2.imread(img_path)
                        if img_array is None:
                            print(f"Warning: Could not read image {img_path}")
                            continue
                        img_array = cv2.resize(img_array, self.img_size)
                        img_array = Image.fromarray(img_array)
                        img_array = self.transform(img_array)
                        features = self.extract_hog_features(
                            np.array(img_array))
                        self.data.append((features, class_num))
                except Exception as e:
                    print(f"Error processing image {img}: {e}")
                    pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label
