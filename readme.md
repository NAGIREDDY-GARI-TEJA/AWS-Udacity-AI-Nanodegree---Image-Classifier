# 🌸 Flower Image Classifier

This project is part of the **AWS AI/ML Scholarship Program** by **Udacity**. It demonstrates the development of an image classifier trained to recognize different types of flowers using a deep learning model.

## 📚 Project Overview

As part of the AWS AI Nanodegree, I built an image classification model that can predict flower species from images. This project showcases the full machine learning pipeline—from data loading and preprocessing to model training, evaluation, and prediction.

## 🛠️ Technologies Used

- Python
- PyTorch
- Jupyter Notebook
- NumPy, Matplotlib, PIL
- torchvision (for models and image transforms)

## 📁 Dataset

The dataset used is the **Oxford 102 Flower Dataset**, which contains images of flowers categorized into 102 different classes. The dataset is split into training, validation, and test sets.

📥 Download link: [Oxford 102 Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## 🚀 Features

- Image preprocessing and augmentation using PyTorch transforms
- Transfer learning using pretrained models like VGG16 or ResNet
- Training and validation loops with accuracy and loss tracking
- Saving and loading model checkpoints
- Predicting top-K flower classes from a new image

## 🔧 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/flower-image-classifier.git
   cd flower-image-classifier

2. Install required dependencies:
   
    ```bash
    pip install -r requirements.txt

3. Train the model:

    ```bash
    python train.py --data_dir flowers --arch vgg16 --epochs 5 --gpu

4. Predict a flower from an image:

    ```bash
    python predict.py image.jpg checkpoint.pth

Results: 

<img width="260" alt="download" src="https://github.com/user-attachments/assets/121105c9-e5f2-4bb1-9c10-2636d5d411d1" />


[0.9950209856033325, 0.0023709421511739492, 0.0011722719063982368, 0.0006429857457987964, 0.0002727250393945724]


['pink primrose', 'pelargonium', 'mallow', 'hibiscus', 'tree mallow']
