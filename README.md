# Skin Lesion Classification Using Convolutional Neural Networks

## Overview

As part of the STEM course in my junior year at Mass Academy, my Independent Research Project explores the use of Convolutional Neural Networks (CNNs) and machine learning for skin lesion classification. The goal is to develop a model capable of analyzing images of skin lesions and classifying them as benign or malignant, demonstrating how deep learning can be applied to medical image analysis and support earlier detection of potentially dangerous skin cancers.

Traditional skin cancer diagnosis often involves biopsies and laboratory analysis, which can take weeks to produce results. This project investigates how machine learning can provide rapid image-based assessments while highlighting both the potential and limitations of AI in healthcare.

## Project Objectives

* Develop a CNN-based image classification model using Python
* Train the model on a publicly available skin lesion dataset
* Classify skin lesion images based on visual characteristics
* Evaluate model performance on unseen test data
* Explore challenges such as image quality, class imbalance, and model generalization

## Dataset

This project uses the HAM10000 (Human Against Machine with 10,000 Training Images) dataset, a widely used collection of dermoscopic skin lesion images containing multiple lesion categories.

Dataset source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Methodology

1. Image preprocessing and normalization
2. Dataset preparation and train/test splitting
3. CNN model design and implementation
4. Model training using labeled skin lesion images
5. Performance evaluation using testing data
6. Analysis of prediction accuracy and model behavior

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Jupyter Notebook

## Results

The model demonstrates the ability of Convolutional Neural Networks to identify patterns in medical images and classify skin lesions based on visual features such as shape, color, and border characteristics.

The project also highlights important challenges in medical AI, including:

* Dataset imbalance
* Overfitting
* Image quality variation
* Generalization to diverse patient populations

## Future Work

* Improve model accuracy through transfer learning
* Increase dataset diversity and representation
* Implement explainable AI techniques such as Grad-CAM
* Develop a user-friendly app interface for image uploads

## Disclaimer

This project was developed only for educational and research purposes. It is not intended to provide medical advice, diagnosis, or treatment and should not be used as a substitute for professional healthcare evaluation.

## Author

Anika Sivasankar

Massachusetts Academy of Math and Science at Worcester Polytechnic Institute
