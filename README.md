#  Image Captioning with CLIP and ViT

This project aims to create a multilingual image captioning system by integrating CLIP and Vision Transformer (ViT) models. It includes steps for downloading images, translating captions, training the model, and generating captions for new images.

## Project Overview

- **Image Downloading:** Download images from URLs and save them locally.
- **Caption Translation:** Translate image captions into multiple languages using Google Translate.
- **Model Training:** Train a multilingual CLIP model with XLM-Roberta and ViT for image captioning.
- **Prediction:** Generate captions for new images and evaluate the model's performance.

## File Structure

- `train.py`: Script for training the multilingual CLIP model.
- `data.py`: Script for translating captions using Google Translate.
- `imagedownload.py`: Script for downloading images from URLs.
- `predict.py`: Script for generating captions for new images and evaluating the model.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- torchvision
- Googletrans
- ijson
- PIL (Pillow)
- requests
- tqdm
- wandb
- scikit-learn
- matplotlib
- pandas

You can install the required packages using:

```bash
pip install torch transformers torchvision googletrans==4.0.0-rc1 ijson pillow requests tqdm wandb scikit-learn matplotlib pandas
