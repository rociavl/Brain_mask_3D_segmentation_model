import os
import torch
import numpy as np
from config.default_config import Config
from data.dataset import BrainSegDataset
from data.transforms import create_transforms
from data.utils import get_paired_files
from models.unet3d import create_model
from models.losses import BinarySegLoss
from training.trainer import Trainer
from training.cross_validation import leave_one_out_cv

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Get pairs of CT files and mask files
    file_pairs = get_paired_files(Config.CT_DIR, Config.MASK_DIR)
    print(f"Found {len(file_pairs)} paired files")

    if len(file_pairs) > 0:
        print(f"Example pair: {file_pairs[0]}")

        # Run leave-one-out cross-validation
        results = leave_one_out_cv(
            file_pairs=file_pairs,
            ct_dir=Config.CT_DIR,
            mask_dir=Config.MASK_DIR,
            num_epochs=Config.NUM_EPOCHS,
            batch_size=Config.BATCH_SIZE
        )

        # Save results
        np.save("loocv_results.npy", results)

if __name__ == "__main__":
    main()