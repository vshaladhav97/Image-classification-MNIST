# Image-classification-MNIST

This project focuses on classifying handwritten digits (0-9) from the MNIST dataset using neural networks. It compares the performance of Fully Connected Networks (FCNs) and Convolutional Neural Networks (CNNs) to identify the optimal architecture for image classification tasks.  

## Dataset Overview  
The MNIST dataset is a benchmark in machine learning, particularly for image classification:  
- **Training Samples**: 60,000 grayscale images  
- **Test Samples**: 10,000 grayscale images  
- **Image Dimensions**: 28×28 pixels  
- **Classes**: 10 (digits 0-9)  

## Project Goals  
- Implement and evaluate multiple neural network architectures.  
- Compare the performance of FCNs and CNNs for image classification.  
- Optimize hyperparameters for improved model accuracy.  

## Methodology  
### 1. Fully Connected Network (FCN)  
- **Input Layer**: 784 neurons (flattened 28×28 image pixels)  
- **Hidden Layers**:  
  - Layer 1: 128 neurons, ReLU activation  
  - Layer 2: 64 neurons, ReLU activation  
- **Output Layer**: 10 neurons (softmax activation for digit classification)  

### 2. Convolutional Neural Network (CNN)  
- **Convolutional Layers**:  
  - Layer 1: 32 filters (3×3), ReLU activation, max pooling  
  - Layer 2: 64 filters (3×3), ReLU activation, max pooling  
- **Fully Connected Layer**: 128 neurons, ReLU activation  
- **Output Layer**: 10 neurons (softmax activation)  

### Training and Validation  
- **Optimizer**: Stochastic Gradient Descent (SGD)  
- **Loss Function**: Cross-Entropy Loss  
- **Hyperparameters**:  
  - Learning Rate: 0.01  
  - Batch Size: 64  
  - Epochs: 5  
- **Validation Set**: MNIST test dataset (10,000 images)  

## Results  
### 1. Fully Connected Network (FCN)  
- **Training Accuracy**: 92.10%  
- **Validation Accuracy**: 92.10%  

### 2. Convolutional Neural Network (CNN)  
- **Training Accuracy**: 99.2%  
- **Validation Accuracy**: 98.5%  

### Key Insights  
- CNNs significantly outperform FCNs for image classification tasks.  
- Convolutional layers capture spatial hierarchies effectively.  
- Hyperparameter optimization improved overall performance.  

## Conclusion  
This project successfully classified MNIST digits and demonstrated the superiority of CNNs in image-related tasks. Future work can include exploring deeper architectures and applying the models to more complex datasets.  

## How to Run  
1. Clone the repository:  
   ```bash
   git clone git@github.com:vshaladhav97/Image-classification-MNIST.git
   ```  
2. Run the training script:  
   ```bash
   python project2.py
   ```  

## Dependencies  
- Python 3.x  
- TensorFlow/Keras or PyTorch  
- NumPy  
- Matplotlib  

## Acknowledgments  
This project was completed as part of my coursework, and I would like to acknowledge the resources and guidance provided during the course.  
