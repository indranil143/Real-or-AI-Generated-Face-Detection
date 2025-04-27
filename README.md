# Real or AI-Generated Face Detection using Deep Learning 🤖👤

## 📌 Project Overview

AI-generated images are everywhere, making it crucial to detect them before misinformation spreads or you get catfished!! 🎭 
Well, this deep learning project uses **Convolutional Neural Networks (CNNs)** to accurately classify images as real human faces or AI-generated ones. By using deep learning techniques, this model enhances our ability to identify synthetic images and mitigate digital deception.
The rise of AI-generated images, particularly those produced by **Generative Adversarial Networks (GANs)** like **StyleGAN**, has blurred the lines between reality and fabrication. These synthetic images are often indistinguishable from real photographs, posing challenges in areas such as **digital security**, **misinformation**, and **privacy**. This CNN-based classifier tackles this challenge by distinguishing real and AI-generated face images.


## 🗂️ Dataset

- **Real Images**: Collected from publicly available facial datasets representing diverse demographics.
- **AI-Generated Images**: Sourced from GAN-based generators producing highly realistic synthetic faces.

💡You can also create your own dataset by collecting images manually. One way is to use the "**Download All Images**" extension in your browser to scrape images from Google’s image search and curate a customized dataset for better model adaptability.

## 🏗️ Model 

Our **CNN architecture** efficiently classifies real and AI-generated faces using multiple convolutional and pooling layers for robust feature extraction.

### 🔧 Model Summary:
- **Input Shape**: (256, 256, 3) RGB images  
- **Conv2D Layers**: Extract spatial features from images  
- **MaxPooling2D Layers**: Reduce dimensionality and retain essential features  
- **Flatten Layer**: Converts feature maps into a 1D vector  
- **Dense Layers**: Process high-level features for final classification  
- **Activation Functions**:  
  - **ReLU** → Used in convolutional layers to introduce non-linearity  
  - **Sigmoid** → Used in the output layer for binary classification  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam 

## 🛠️ Installation

To set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/indranil143/Real-or-AI-Generated-Face-Detection.git

2. **Navigate to the Directory**:
   ```bash
   cd Real-or-AI-Generated-Face-Detection

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

## Usage
To classify an image:

- **Prepare Your Image dataset**: Ensure the image is in a supported format (e.g., **JPEG**, **PNG**) and appropriately preprocessed.​ 
- **Run the Classifier**: Use the provided script or notebook to input your image and receive the classification result.​

## Results
Here’s an example output from the model:
![Example Result](https://github.com/indranil143/Real-vs.-AI-Generated-Face-Classification-using-Deep-Learning/blob/main/Screenshot_example_result.jpeg)

The trained model demonstrates **high accuracy** in distinguishing between real and AI-generated face images. 
Although, to ensure **generalization**, testing on unseen data and applying **regularization techniques** like dropout and data augmentation are recommended.

## License
This project is licensed under the **MIT License**. See the LICENSE file for more information.
