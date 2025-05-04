import os
import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose

class BrainSegDataset(Dataset):
    def __init__(self, ct_path, mask_path, file_pairs, transform=None):
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # Get file paths
        ct_file, mask_files = self.file_pairs[idx]
        ct_file_path = os.path.join(self.ct_path, ct_file)
        ct_data, _ = nrrd.read(ct_file_path)
        ct_tensor = torch.from_numpy(ct_data).float().unsqueeze(0)

        # Create a binary mask by thresholding the average of all masks
        if isinstance(mask_files, list):
            masks_data = []
            for mask_file in mask_files:
                mask_file_path = os.path.join(self.mask_path, mask_file)
                mask_data, _ = nrrd.read(mask_file_path)
                masks_data.append(mask_data)

            mask_data = np.mean(masks_data, axis=0)
            mask_data = (mask_data > 0.5).astype(np.float32)
        else:
            mask_file_path = os.path.join(self.mask_path, mask_files)
            mask_data, _ = nrrd.read(mask_file_path)
            mask_data = (mask_data > 0.5).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0)

        data = {"image": ct_tensor, "mask": mask_tensor}

        if self.transform:
            transformed_data = self.transform(data)
            if isinstance(transformed_data, list):
                image = transformed_data[0]["image"]
                mask = transformed_data[0]["mask"]
            else:
                image = transformed_data["image"]
                mask = transformed_data["mask"]
            return image, mask

        return data["image"], data["mask"]