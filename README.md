# Generative AI Empowered Skin Cancer Diagnosis: A Classification through Deep Learning

This project focuses on skin lesion classification using deep learning techniques, specifically Generative Adversarial Networks (GANs) for synthetic data augmentation and Convolutional Neural Networks (CNNs) for classification. The objective is to enhance diagnostic accuracy by increasing the diversity and volume of training data through synthetic image generation.

## Introduction
Skin cancer is one of the most common forms of cancer, and early detection is crucial for effective treatment. This project aims to build a robust skin lesion classification system by combining the power of GANs for synthetic image generation and CNNs for accurate classification.

## Features
- **Synthetic Data Generation**: Using GAN to create additional data from limited datasets.
- **CNN-based Classification**: A deep learning model trained to classify skin lesions into categories such as benign and malignant.
- **Data Augmentation**: Enhance the dataset by applying transformations and synthetic image generation.
- **Exploratory Data Analysis (EDA)**: Analyze the dataset with visualizations and insights.
- **Performance Evaluation**: Accuracy, precision, recall, and F1-score metrics for evaluating the model.

## Dataset
We use the **HMNIST 28x28 RGB dataset** for skin lesion classification. The original dataset is augmented with synthetic images generated using GAN to increase the training data diversity.

- **Original Dataset**: [Kaggle - hmnist_28_28_RGB.csv](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- **Generated Data**: 1000 synthetic images generated using GAN over 200 epochs.

## Model Architecture
- **Generative Adversarial Network (GAN)**: Used for synthetic data generation.
- **Convolutional Neural Network (CNN)**: Used for the classification of skin lesions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/synthetic-data-skin-lesion-classification.git
    cd synthetic-data-skin-lesion-classification
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment (e.g., Jupyter Notebook or Google Colab).

## Usage

1. **Data Preprocessing**:
    - Load the dataset and preprocess it for training the GAN and CNN models.

2. **Training the GAN**:
    - Train the GAN model to generate synthetic skin lesion images.
    ```bash
    python train_gan.py
    ```

3. **Training the CNN**:
    - Train the CNN model using the augmented dataset (original + synthetic data).
    ```bash
    python train_cnn.py
    ```

4. **Testing the Model**:
    - Evaluate the performance of the model on the test dataset.
    ```bash
    python evaluate.py
    ```

5. **Visualization**:
    - View results of the classification and generated images.
    ```bash
    python visualize_results.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
