# Audio MNIST Digit Recognition

## Introduction
This project focuses on building and comparing deep learning models for classifying spoken digits from a set of audio recordings. It addresses the core task of converting raw audio into a machine-readable format and leveraging neural network architectures to accurately recognize the numerical value of the spoken word. The final model demonstrates the effectiveness of deep learning in audio classification, a key area of machine learning.

## Problem Statement
The primary objective of this project is to develop an accurate model that can correctly identify a spoken digit (0-9) from an audio file. The main challenges addressed are:
1.  **Feature Extraction**: Transforming raw, one-dimensional audio data into meaningful, structured features that deep learning models can interpret effectively.
2.  **Model Selection**: Training and evaluating multiple deep learning models to determine the most suitable architecture for the classification task.
3.  **Performance Evaluation**: Achieving a high level of accuracy and validating the model's performance using appropriate metrics.

## Dataset Details
The project utilizes the **Audio MNIST Archive**, a publicly available dataset.
* **Source**: The data is a zip file from an online archive.
* **Data Size**: The dataset contains a total of **5,000** audio files.
* **Class Distribution**: The dataset is perfectly balanced, with **500** audio files for each digit from 0 to 9.
* **File Format**: All audio files are in `.wav` format.
* **Data Preparation**: The audio files were extracted from a compressed archive and loaded using the `librosa` library. The dataset was then split into training and testing sets to evaluate model performance.

## Methods and Algorithms
A multi-stage methodology was followed to build and validate the models.
* **Audio Feature Extraction**: The `librosa` library was used to process each audio file. Mel-Frequency Cepstral Coefficients (**MFCCs**) were calculated to represent the unique frequency characteristics of each spoken digit. The MFCCs were then converted into **spectrograms**, which represent the audio signals as 2D images, making them suitable for convolutional neural networks.
* **Model Training**: Two deep learning models were trained:
    * **Artificial Neural Network (ANN)**: A multi-layered perceptron was used as a baseline classification model.
    * **Convolutional Neural Network (CNN)**: A CNN was employed to effectively process the spectrograms as 2D image data, taking advantage of its ability to identify spatial patterns.
* **Model Evaluation**: The performance of each model was measured by its **accuracy**, and detailed performance for each digit was analyzed using a **classification report**, which includes precision, recall, and F1-score.

## Key Results
* **High Accuracy**: Both the ANN and CNN models achieved excellent performance. The **ANN model** achieved an accuracy of **99.4%**, while the **CNN model** reached an even higher accuracy of **99.8%**.
* **Model Justification**: While both models were highly effective, the **CNN was chosen as the superior model**. This decision was based on its architectural design, which is specifically optimized to recognize patterns within the 2D spectrogram representation of the audio data. This approach is more robust and scalable for similar audio processing tasks.

## Tech Stack
* **Libraries**:
    * `librosa` and `IPython.display.Audio` for audio file loading, visualization, and feature extraction.
    * `pandas`, `numpy`, and `os` for data manipulation, array operations, and file system navigation.
    * `matplotlib` and `seaborn` for visualizing audio waveforms and spectrograms.
    * `scikit-learn` for data splitting and model evaluation metrics.
    * `tensorflow.keras` for building and training the deep learning models.