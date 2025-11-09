# Brain Mask 3D Segmentation Model 🧠🩻

This repository contains a deep learning model for automatic 3D brain region segmentation from post-surgery CT scans. It supports downstream tasks like SEEG electrode localization and path reconstruction.

## Problem
Post-operative CT scans contain brain tissue with foreign elements like SEEG electrodes. Segmenting the brain helps isolate relevant anatomy for subsequent electrode detection.

## Methodology
- Framework: MONAI + PyTorch
- Model: 3D UNet
- Data: Post-operative CT scans from epilepsy patients
- Features:
  - CT normalization
  - Skull and bolt artifact removal (optional)
  - Dice + CrossEntropy loss
  - Patch-based training for GPU efficiency

## 📊 Results
Achieved high overlap (Dice > 0.85) on test cases. Qualitative results show precise brain contouring despite bolt interference.

![image](https://github.com/user-attachments/assets/7aa8fd1a-0499-4ec4-a42d-ed156455d835)

## 🎥 Video Demo
Watch the brain segmentation model in action:

[![Brain Segmentation Demo](https://img.youtube.com/vi/6vQeFYLTGLA/maxresdefault.jpg)](https://youtu.be/6vQeFYLTGLA)

## 🚀 Live Demo
You can try the model directly in your browser on Hugging Face Spaces: [seeg-brain-mask-segmentationn](https://huggingface.co/spaces/rocioavl/seeg-brain-mask-segmentation).


## Author

**Rocío Ávalos**  
Biomedical Engineer @ Universitat Politècnica de Catalunya  
Intern @ Hospital del Mar | Research @ Center for Brain and Cognition  
[LinkedIn](https://www.linkedin.com/in/rocioavalos) • [GitHub](https://github.com/rociavl)


citation
MONAI: An open-source framework for deep learning in healthcare. https://doi.org/https://doi.org/10.48550/arXiv.2211.02701
