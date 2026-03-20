# MNIST Digit Classifier 🔢

A Convolutional Neural Network (CNN) built with PyTorch that classifies handwritten digits (0–9) from the MNIST dataset. Trained and run on Google Colab with GPU support.

---

## Overview

This project trains a CNN on the MNIST dataset to recognize handwritten digits. After training, the model can accept a user-uploaded image and predict the digit it contains.

---

## Features

- CNN architecture with two convolutional blocks and fully connected layers
- GPU acceleration via CUDA (falls back to CPU automatically)
- Data normalization and augmentation pipeline
- Model persistence — save and reload weights from Google Drive
- Image prediction from any uploaded grayscale image

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework and transforms |
| `pandas` | Loading the CSV dataset |
| `numpy` | Numerical operations |
| `matplotlib` | Visualizing images and results |
| `scikit-learn` | Train/test split |
| `Pillow` | Loading and preprocessing input images |

Install all dependencies:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn Pillow
```

---

## Dataset

The project uses the [MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer/data) in CSV format (`train.csv`), where:

- Each row represents one image
- The first column (`label`) is the digit class (0–9)
- The remaining 784 columns are pixel values (28×28 image, flattened)

Place `train.csv` in your Google Drive at `MyDrive/train.csv` before running.

---

## Model Architecture

```
Input: (batch, 1, 28, 28)
  → Conv2d(1→32, 3×3, padding=1) + ReLU + MaxPool → (batch, 32, 14, 14)
  → Conv2d(32→64, 3×3, padding=1) + ReLU + MaxPool → (batch, 64, 7, 7)
  → Flatten → (batch, 3136)
  → Linear(3136→128) + ReLU
  → Linear(128→10)
Output: logits for 10 digit classes
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 5 |
| Train/Test split | 80% / 20% |

---

## Usage

### 1. Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive/')
```

### 2. Train the Model

Run all cells sequentially. Training output will look like:

```
Epoch 1/5 → Loss: 0.1523 → Accuracy: 95.42%
Epoch 2/5 → Loss: 0.0487 → Accuracy: 98.61%
...
Training Complete! 🎉
```

### 3. Evaluate on Test Set

The model is evaluated automatically after training:

```
Test Accuracy: 99.05%
```

### 4. Save & Load Model

The trained model weights are saved to Google Drive:

```python
# Save
torch.save(model.state_dict(), '/content/drive/MyDrive/mnist_model.pth')

# Load
model.load_state_dict(torch.load('/content/drive/MyDrive/mnist_model.pth'))
```

### 5. Predict a Custom Image

Upload any handwritten digit image when prompted. The model will:
1. Convert it to grayscale
2. Invert colors (expects white digit on black background)
3. Resize to 28×28
4. Normalize and run inference

```python
prediction = predict_digit('your_image.png')
print(f'Model predicts: {prediction}')
```

---

## File Structure

```
MyDrive/
├── train.csv           # MNIST dataset (input)
└── mnist_model.pth     # Saved model weights (output)
```

---

## Notes

- The `predict_digit()` function **inverts** the uploaded image, so it works best with **dark digits on a light background** (e.g., written on white paper).
- The model achieves ~99% test accuracy after just 5 epochs.
- Google Colab file upload cells (`files.upload()`) are Colab-specific and won't run in a standard Python environment — replace with a local file path if running locally.

---

## License

This project is for educational purposes. The MNIST dataset is publicly available via [Yann LeCun's website] and Kaggle.
