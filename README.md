# Cats and Dogs Breed Classification – Oxford-IIIT Pet Dataset

This project focuses on the classification of dog and cat breeds using the Oxford-IIIT Pet Dataset. It includes a complete end-to-end pipeline: data preparation, visualization, model training, evaluation, and performance analysis using both a custom CNN and transfer learning with ResNet18.

## Dataset

- **Source**: [Kaggle – Cats and Dogs Breeds Classification (Oxford Dataset)](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)
- **Content**: 7,390 labeled images across 37 distinct dog and cat breeds.

## Project Structure

### Data Preparation

- Automatic download and extraction of the dataset from Kaggle.
- Filenames are parsed to extract class labels.
- Images are split into training (70%), validation (10%), and test (20%) sets, with a balanced class distribution.

### Visualization

- Displays random sample images from each data split.
- Analyzes class distribution using bar plots and tabular summaries.

### Dataset Loader

- Custom `EncodedDataset` class built using PyTorch `Dataset` and `DataLoader`.
- Applies image transformations including resizing, normalization, and data augmentation.

### Model Training

- Two architectures implemented:
  1. Custom Convolutional Neural Network (CNN).
  2. ResNet18 with transfer learning.
- Training is performed using the Adam optimizer and CrossEntropyLoss.
- Validation metrics include accuracy, precision, recall, and F1-score.

### Evaluation

- Evaluates model performance on validation and test sets.
- Generates confusion matrices for detailed error analysis.
- Visual comparison of predicted vs. true class labels.

## Installation

Install dependencies with:

```bash
pip install kaggle matplotlib torchvision scikit-learn torch pandas seaborn tqdm
