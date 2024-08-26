# CropsInjuryDetection

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Installation](#4-installation)
   1. [Dependencies](#41-dependencies)
   2. [Setting up Environment](#42-setting-up-environment)
5. [Model Details](#5-model-details)
6. [Evaluation](#6-evaluation)
7. [Usage](#7-usage)
8. [License](#8-license)

## 1. Introduction

Welcome to the "CropsInjuryDetection" project! This project was developed to classify vegetation into four distinct classes: Blight, Common Rust, Gray Leaf Spot, and Healthy. The goal is to assist in the early detection of diseases in crops, allowing for more timely and effective interventions.

The project was implemented in a Jupyter Notebook designed to run on Google Colab. It utilizes a Convolutional Neural Network (CNN) developed using PyTorch to achieve high classification accuracy.

## 2. Dataset

The dataset used for this project can be downloaded from the following link: [Crops Injury Detection Dataset](https://drive.google.com/file/d/1pXjbbvxFMctaZvfSY3E0fn0nMa0TB7G0/view).

### Important:
- After downloading, place the dataset in the same directory as the notebook `Detection_of_diseases_in_crops.ipynb`.

## 3. System Architecture

The project employs a Convolutional Neural Network (CNN) to classify images into one of four classes:
- Blight
- Common Rust
- Gray Leaf Spot
- Healthy

The architecture is simple yet effective, making it suitable for deployment on various platforms, including mobile and embedded systems.

## 4. Installation

To set up and run the project, follow these steps:

### 4.1 Dependencies

Make sure you have the following dependencies installed:

- PyTorch
- NumPy

### 4.2 Setting up Environment

Since the project is designed to run on Google Colab, setting up the environment is straightforward:

1. Clone the repository or upload the `Detection_of_diseases_in_crops.ipynb` notebook to your Google Drive.
2. Download the dataset and place it in the same directory as the notebook.
3. Open the notebook in Google Colab.
4. Install the required libraries by running the setup cells in the notebook.

## 5. Model Details

The CNN model used in this project is designed to process images of crops and classify them into the four categories mentioned above. Below is the structure of the model:

```python
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=131072, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x
```

**Key Components**:

- Conv Layers: Extract features from the input images.
- ReLU Activation: Introduces non-linearity to the model.
- MaxPooling: Reduces the spatial dimensions of the feature maps.
- Fully Connected Layers: Combine the features and map them to the output classes.

## 6. Evaluation

The model's performance was evaluated using accuracy and a confusion matrix. The confusion matrix helps in understanding the model's classification performance across all four classes.

## 7. Usage
To use the model:

1. Ensure the dataset is in the correct directory.
2. Open the Detection_of_diseases_in_crops.ipynb notebook in Google Colab.
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

Example:
Run the cells in the notebook to train the CNN model and classify a batch of images from the dataset.

## License

This project is licensed under the [MIT License](LICENSE).
