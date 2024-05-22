import cv2
import os
import csv
import torch
import numpy as np
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST


### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - Classification                                                                                            ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Given a greyscale image with ten bright dots, the user will find the ROIs in a different function and store the information in a different file
# we want to first get all the file paths needed 
master_file_path = r""

# fiel path using MacOS
image_path = os.path.join(master_file_path, "ROI.tif")
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print("Error: Image not found.")
    exit()

### ---------------------------------------------------------------------------------------------------------------- ###
### Function Definitions                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Function to load image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Image not found.")
        exit()
    return image


### ---------------------------------------------------------------------------------------------------------------- ###
### Main                                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###

# loading the images in using threadpool
image_path = os.path.join(master_file_path, "ROI.tif")
# load all the images used to training the final mapping matrix
with concurrent.futures.ThreadPoolExecutor() as executor:
    image = executor.submit(load_image, image_path)

def generate_dataset_for_digit(digit, num_samples):
    X = []
    y = []
    mnist = MNIST('dataset')  # Assuming you have MNIST dataset downloaded
    mnist_images, mnist_labels = mnist.load_training()
    mnist_images = np.array(mnist_images)
    mnist_labels = np.array(mnist_labels)
    digit_indices = np.where(mnist_labels == digit)[0]
    for _ in range(num_samples):
        image = load_image(image_path)
        spots = np.random.randint(50, 256, size=14)  # Generate 14 random brightness values
        X.append(spots)  # Add spots as input vector
        # Map the image to the specified digit
        random_mnist_index = np.random.choice(digit_indices)
        mnist_image = mnist_images[random_mnist_index].reshape(28, 28)
        mnist_label = mnist_labels[random_mnist_index]
        y.append(mnist_label)
    return np.array(X), np.array(y)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(14, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X_train, y_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Train your supervised learning model for each digit separately
num_samples_per_digit = 1000  # You can adjust this number as needed
for digit in range(10):
    print(f"Generating dataset for digit {digit}...")
    X_digit, y_digit = generate_dataset_for_digit(digit, num_samples_per_digit)
    print(f"Training model for digit {digit} with {num_samples_per_digit} samples...")
    train_model(X_digit, y_digit)
    