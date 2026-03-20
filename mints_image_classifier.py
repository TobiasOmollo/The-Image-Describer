# -*- coding: utf-8 -*-
"""Mints_Image_Classifief.ipynb

Original file is located at
    https://colab.research.google.com/drive/**********
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/content/drive/MyDrive/train.csv')
print(df.head())

print(df.shape)
print(df.isnull().sum().sum())

#Seperating X From Y
# Y = label column (what the digit it is)
Y = df['label'].values

# X = all columns except label (pixel values)
X = df.drop('label', axis=1).values

print("X shape:", X.shape)
print("\n", "="*50)
print("Y shape:", Y.shape)
print("\n","="*50)

# Normalize pixel values from 0-255 to 0-1
X = X / 255.0

# Reshape from (42000, 784) to (42000, 1, 28, 28)
X = X.reshape(-1, 1, 28, 28)

print("X shape after reshape:", X.shape)

#Confirming our Data correctness visually
# Visualize a sample image
plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {Y[0]}')
plt.show()

#Data Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("="*50)
print("Testing size:", X_test.shape)
print("="*50)

#Pytorch Tensors
#NOTE⨷💀⚡: Tensors work with CPU and GPU combined
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test  = torch.tensor(Y_test,  dtype=torch.long)

# Check if GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#Data Loader
from torch.utils.data import TensorDataset, DataLoader

# Create datasets
train_dataset = TensorDataset(X_train, Y_train)
test_dataset  = TensorDataset(X_test,  Y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print("Total training batches:", len(train_loader))
print("Total testing batches:",  len(test_loader))

#The CNN Architecture
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        # Stage 1: First Convolution Block
        self.conv1 = nn.Conv2d(
            in_channels=1,   # grayscale = 1 channel
            out_channels=32, # 32 filters
            kernel_size=3,   # 3x3 filter
            padding=1        # keeps image same size
        )

        # Stage 2: Second Convolution Block
        self.conv2 = nn.Conv2d(
            in_channels=32,  # receives 32 channels
            out_channels=64, # outputs 64 filters
            kernel_size=3,
            padding=1
        )

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2) # 2x2 window

        # Activation Function
        self.relu = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 outputs

    def forward(self, x):
        # Block 1
        x = self.conv1(x)  # Convolve
        x = self.relu(x)   # Activate
        x = self.pool(x)   # Pool → 14x14

        # Block 2
        x = self.conv2(x)  # Convolve
        x = self.relu(x)   # Activate
        x = self.pool(x)   # Pool → 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully Connected
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Initialize model and move to GPU
model = MNISTModel().to(device)
print(model)

#Calling the Loss Function and the Otimizer
# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    #Testng accuracy
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{epochs} → '
          f'Loss: {running_loss/len(train_loader):.4f} → '
          f'Accuracy: {accuracy:.2f}%')

print("Training Complete! 🎉")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')


# Saving the model to Google Drive for easy access
torch.save(model.state_dict(),
           '/content/drive/MyDrive/mnist_model.pth')
print("Model saved to Google Drive! 🎉")
print("="*50)
print(" And Model saved successfully! 🎉")

# To Load the model in the future
# (Drive is already mounted above)
model = MNISTModel().to(device)
model.load_state_dict(torch.load(
    '/content/drive/MyDrive/mnist_model.pth'))
model.eval()
print("Model loaded successfully! To Drive🎉")

#Setting up Input space
"""Where we will upload a picture containing a digit to 
    for the model to predict"""

#The Predict digit function
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import numpy as np

def predict_digit(image_path):
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)

    if np.mean(img_array) > 127:
        image = ImageOps.invert(image)

    img_array = np.array(image)

    rows = np.any(img_array > 50, axis=1)
    cols = np.any(img_array > 50, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)

        image = Image.fromarray(img_array[rmin:rmax, cmin:cmax])

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

#The Input Loop
from google.colab import files

while True:
    print("\n" + "="*40)
    print("Upload an image to predict")
    print("(Close/Cancel upload to stop)")
    print("="*40)
    
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Stopping! 👋")
        break
    
    image_path = list(uploaded.keys())[0]
    
    prediction = predict_digit(image_path)
    print(f'Model predicts: {prediction}')
    
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Digit: {prediction}')
    plt.show()
    
    another = input("\nTest another image? (yes/no): ")
    if another.lower() != 'yes':
        print("Thanks for testing! 👋")
        break

