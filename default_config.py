import os
import torch

class Config:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths
    CT_DIR = "/content/drive/MyDrive/TFG 💪🧠/Code/Modelos /Brain_mask_model/Dataset/CT"
    MASK_DIR = "/content/drive/MyDrive/TFG 💪🧠/Code/Modelos /Brain_mask_model/Dataset/MASK"
    
    # Training parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    CHANNELS = (32, 64, 128, 256, 512)
    STRIDES = (2, 2, 2, 2)
    NUM_RES_UNITS = 2
    DROPOUT = 0.5
    
    # Loss function weights
    DICE_WEIGHT = 0.8
    BCE_WEIGHT = 0.2
    
    # Post-processing
    MIN_REGION_SIZE = 100
    THRESHOLD = 0.5
    
    # Visualization
    OUTPUT_DIR = "predictions_2_brain_mask"
    
    # Random seeds
    SEED = 42