# GenerateTextFromImage Documentation

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

Welcome to the documentation for the "GenerateTextFromImage" project. This project focuses on generating textual descriptions for images using deep learning techniques. The project was developed on Google Colab and uses a combination of convolutional neural networks (CNN) and natural language processing (NLP) to achieve the desired outcomes. The primary model uses the VGG16 architecture for feature extraction from images, and these features, combined with image descriptions, are used to train the main model.

## 2. Dataset

The dataset used for this project is the Flickr8k dataset, which contains 8,000 images along with five different captions for each image. You can download the dataset from the following link: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k/data).

### Important:
- After downloading, place the dataset in the same directory as the Jupyter notebook `GenerateTextImage.ipynb`.

## 3. System Architecture

The project utilizes a modular architecture, where image features are extracted using the VGG16 model. These features are then passed through a deep learning model to generate descriptive text. The model architecture integrates both convolutional layers (for image processing) and recurrent layers (for text generation).

## 4. Installation

To set up the project, follow these steps:

### 4.1 Dependencies

Make sure you have the following dependencies installed:

- TensorFlow
- Keras
- Numpy
- Pandas
- Python 3.10.13

### 4.2 Setting up Environment

Since the project was developed on Google Colab, the environment setup is straightforward:

1. Clone the repository or upload the `GenerateTextImage.ipynb` file to your Google Drive.
2. Download the dataset and place it in the same directory as the notebook.
3. Open the notebook in Google Colab.
4. Install the required libraries by running the setup cells in the notebook.

## 5. Model Details

The model used in this project consists of two primary components:

1. **VGG16**: This pre-trained CNN model is used for feature extraction from the input images. The extracted features represent the essential visual information from the images.

2. **Text Generation Model**: The main model takes the extracted image features and combines them with the corresponding text descriptions to learn a mapping between images and textual descriptions. The structure of the text generation model is as follows:

   ```python
   input_1 = Input(shape=(4096,))
   drop_1 = Dropout(0.45)(input_1)
   dense_1 = Dense(256, activation='relu')(drop_1)  # Layer responsible for text processing

   input_2 = Input(shape=(max_length,))
   seq = Embedding(vocab_size, 256, mask_zero=True)(input_2)
   drop_2 = Dropout(0.45)(seq)
   lstm = LSTM(256)(drop_2)

   decoder1 = add([dense_1, lstm])
   decoder2 = Dense(256, activation='relu')(decoder1)
   outputs = Dense(vocab_size, activation='softmax')(decoder2)

   model = Model(inputs=[input_1, input_2], outputs=outputs)
   model.compile(loss='categorical_crossentropy', optimizer='adam')
``

** Key Components**:
- input_1: Takes the features extracted by VGG16.
- input_2: Takes the sequence of words representing the description.
- Dropout layers: Used to prevent overfitting.
- Dense and LSTM layers: Used to process the features and generate the final description.
- Output layer: Generates the probability distribution over the vocabulary to form the final caption.

## 6. Evaluation

The model's performance was evaluated using accuracy and a confusion matrix. The confusion matrix helps in understanding the model's classification performance across all four classes.

## 7. Usage
To use the model:

1. Ensure the dataset is in the correct directory.
2. Open the Detection_of_diseases_in_crops.ipynb notebook in Google Colab.
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

Example:
Run the cells in the notebook to train the CNN model and classify a batch of images from the dataset.

## 8. License
This project is licensed under the [MIT License](LICENSE).
