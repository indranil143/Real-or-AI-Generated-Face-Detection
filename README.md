# Real vs. AI-Generated Face Classification using Deep Learning 🤖👤

## 📌 Project Overview

AI-generated images are now trending or can say everywhere, and spotting them before you get catfished is more important than ever!! 🎭 
Well, this deep learning project harnesses the power of Convolutional Neural Networks (CNNs) to separate authentic human faces from AI-generated illusions with razor-sharp accuracy.
This project leverages the power of Convolutional Neural Networks (CNNs) to accurately classify images as either real human faces or AI-generated facsimiles. By employing advanced deep learning techniques, this model aims to enhance our ability to detect and mitigate the spread of deceptive digital content.
The rise of AI-generated images, particularly those produced by Generative Adversarial Networks (GANs) like StyleGAN, has blurred the lines between reality and fabrication. These synthetic images are often indistinguishable from real photographs, posing challenges in areas such as digital security, misinformation, and privacy. This project addresses these challenges by developing a CNN-based classifier capable of differentiating between real and AI-generated face images.


## 🗂️ Dataset

- **Real Images**: Sourced from publicly available facial datasets capturing diverse demographics.
- **AI-Generated Images**: Produced using advanced GAN architectures and other AI image generators that create highly realistic synthetic faces.

You can also create your own dataset by collecting images manually. One way is to use the "Download All Images" extension in your browser to scrape images from Google’s image search. This allows you to curate a custom dataset tailored to your specific needs, improving the model's adaptability.

## 🏗️ Model 

The Convolutional Neural Network (CNN) used in this project is designed to effectively classify real and AI-generated face images. The model follows a sequential architecture with multiple convolutional and pooling layers, ensuring efficient feature extraction and classification.

### 🔧 Model Summary:
- **Input Shape**: (256, 256, 3) RGB images
- **Conv2D Layers**: Extracts spatial features from images
- **MaxPooling2D Layers**: Reduces dimensionality and retains essential features
- **Flatten Layer**: Converts feature maps into a 1D vector
- **Dense Layers**: Processes high-level features for final classification
- **Activation Functions**:
  -- **ReLU**: Used in convolutional layers to introduce non-linearity
  -- **Sigmoid**: Used in the output layer for binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## 🛠️ Installation

To set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/indranil143/Real-vs.-AI-Generated-Face-Classification-using-Deep-Learning.git

2. **Navigate to the Directory**:
   ```bash
   cd Real-vs.-AI-Generated-Face-Classification-using-Deep-Learning

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

## Usage
To classify an image:

- **Prepare Your Image dataset**: Ensure the image is in a supported format (e.g., JPEG, PNG) and appropriately preprocessed.​ 
- **Run the Classifier**: Use the provided script or notebook to input your image and receive the classification result.​

## Results
The trained model demonstrates high accuracy in distinguishing between real and AI-generated face images. 
Although, to ensure generalization, testing on unseen data and applying regularization techniques like dropout and data augmentation are recommended.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

