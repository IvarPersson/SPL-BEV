import torch
import cv2
from torch.utils.data import Dataset

class BEVDataset(Dataset):
    def __init__(self, images, image_dir= "./data/mini/"):
        self.image_dir = image_dir
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = self.image_dir + image['file_name']
        image_vals = cv2.imread(image_path).transpose(2, 0, 1)
        camera_matrix = torch.FloatTensor(image["camera_matrix"])
        dist_poly = torch.FloatTensor(image['dist_poly'])
        return torch.FloatTensor(image_vals/255), dist_poly, camera_matrix,\
            idx, image['file_name'][:-4]