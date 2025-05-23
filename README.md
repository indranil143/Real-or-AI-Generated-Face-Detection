# Deepfake Face Detection using Deep Learning 🤖👤
## (Real-or-AI-Generated-Face-Detection) 

## 📌Project Overview
AI-generated images are now almost everywhere, making it crucial to detect them before misinformation spreads or you get catfished!! 🎭 
In the era of increasingly realistic synthetic media, being able to accurately identify AI-generated content is becoming vital for combating misinformation and maintaining digital trust. 
This repository showcases approaches to classifying facial images as either real or AI-generated (deepfake). The primary focus is on a robust deep learning model utilizing transfer learning with the powerful Xception network and K-Fold cross-validation. An alternative approach using a custom Convolutional Neural Network is also documented.

## 👤 Problem Statement

The rapid advancement in generative adversarial networks (GANs), Flickr and other deep learning techniques has made it increasingly difficult to discern authentic facial images from synthetic ones. Malicious use of deepfakes poses significant threats, including the spread of disinformation, reputational damage, and erosion of trust in digital media. This project explores different deep learning models to help identify AI-generated face images.

## 🚀 Main Approach: Xception-based Transfer Learning with K-Fold CV

The primary and more advanced approach implemented in this repository utilizes a sophisticated transfer learning method with the Xception model and robust K-Fold cross-validation. The detailed implementation and code for this approach are found in the notebook `Deepfake-Face-Detection-with-Xception.ipynb`.

### Approach Details

The solution employs a robust deep learning workflow for training and evaluating the model, implemented in the accompanying Jupyter Notebook (`Deepfake-Face-Detection-with-Xception.ipynb`):

1.  **Data Loading & Preprocessing:** Images are loaded from 'real' and 'fake' directories, resized to **(224, 224) pixels**, normalized (0-1 range), and labeled for binary classification, with basic error handling.
2.  **Data Augmentation:** On-the-fly data augmentation is applied during training, including horizontal flips, brightness, contrast, saturation, and hue adjustments, to improve generalization.
3.  **Model Architecture:** Utilizes the **Xception** convolutional neural network (pre-trained on ImageNet) as a base, with its top layer removed. A custom binary classification head is added, including Global Average Pooling, **Dense layers with L2 regularization (0.0001)** and **Dropout (rate 0.3)**, Batch Normalization, and a final `sigmoid` output layer.
4.  **Two-Phase Training Strategy:** Training is conducted in two phases using the Adam optimizer:
    * **Phase 1:** Train the custom head with the Xception base **frozen** for **20 epochs** with an **initial learning rate of 0.001**.
    Early Stopping (patience 10) and ReduceLROnPlateau (patience 4) callbacks are used.
    * **Phase 2:** **Fine-tune** the entire model (unfreeze Xception base) for an **additional 30 epochs** with a **lower learning rate of 1e-5**,
    Updated callback patience (Early Stopping 15, ReduceLROnPlateau 7) are used.
5.  **K-Fold Cross-Validation:** **5-Fold Cross-Validation** (`SEED = 42`) is implemented to train and evaluate the model 5 times on different data splits, providing a more robust performance estimate.
   Model checkpoints are saved for each fold.
7.  **Evaluation:** Performance is rigorously evaluated using key metrics like Accuracy, Precision, Recall, F1-Score, and AUC. **Classification Reports** and **Confusion Matrices** are generated from pooled predictions across all folds.
   **Training History Plots** and **ROC Curves** are also visualized.
9.  **Prediction:** Includes functionality to load saved models (`.keras` file) and make predictions on new images or an entire directory of images, using a **0.5 probability threshold** to determine the final class.

## 🗂️ Dataset

This project trains and evaluates the main model on the **RVF10K dataset**.

* **Dataset Name:** RVF10K
* **Source:** Kaggle - `sachchitkunichetty/rvf10k`
* **Structure:** Expected to contain `train/real` and `train/fake` subdirectories at the path `/kaggle/input/rvf10k/rvf10k/train`.
* **Size:** Contains approximately 7000 images (split between real and fake).

**Some example images -** ![some image examples](https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20194643.png)
## ✅ Results (Main Approach)

The model was trained using the defined K-Fold cross-validation strategy on the RVF10K dataset. Below is the summary of the evaluation metrics obtained from the notebook output:
<div align="center">

**K-Fold Cross-Validation Summary:**

| Metric            | Value           |
|-------------------|-----------------|
| Average Accuracy  | 0.7840 ± 0.0102 |
| Average AUC       | 0.8609 ± 0.0106 |

**Overall Metrics (Pooled Predictions Across All Folds):**

| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0.0 (Real)   | 0.79      | 0.77   | 0.78     | 3500    |
| 1.0 (Fake)   | 0.77      | 0.80   | 0.79     | 3500    |
| **Accuracy** |           |        | **0.78** | 7000    |
| Macro Avg    | 0.78      | 0.78   | 0.78     | 7000    |
| Weighted Avg | 0.78      | 0.78   | 0.78     | 7000    |

</div>

**Evaluation Plots:**

Visualizations provide further insight into the model's performance. Include the generated plots here:

* **Overall Confusion Matrix:** Shows the counts of True Positives, True Negatives, False Positives, and False Negatives based on pooled predictions.
    <p align="center">
    <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20194747.png" alt="Overall Confusion Matrix" width="350"/>
    </p>

* **Overall ROC Curve:** Illustrates the model's ability to discriminate between classes across different thresholds.
    <p align="center">
    <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20194855.png" alt="Overall Confusion Matrix" height= "400", width="500"/>
    </p>

**Prediction Examples:**

* **Example 1:** <p align="center"> <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20195030.png" alt="Example 1 Prediction" width="1200"/></p>
    * **Filename:** `0M0F4YJ1G9.jpg`
    * **Prediction:** Fake
    * **Probability (Fake):** 0.9294
    * *Comment:* A correct prediction with high confidence that the image is Fake.

* **Example 2:** <p align="center"> <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20195118.png" alt="Example 2 Prediction" width="1300"/> </p>
    * **Filename:** `00292.jpg`
    * **Prediction:** Real
    * **Probability (Fake):** 0.0561
    * *Comment:* A correct prediction with high confidence that the image is Real.

---

## 🔄 Alternative Approach: Custom CNN Model

This section details an earlier project exploring a custom-built Convolutional Neural Network (CNN) for the same real vs. AI-generated face classification task. This serves as a comparison to the more advanced transfer learning approach with Xception. The implementation for this approach is found in the notebook `Real vs. AI-Generated Face Classification model.ipynb`.

This project's core is a custom Convolutional Neural Network architecture designed for binary image classification.

### Model Summary:
* **Input Shape**: The model is designed to accept images with an input shape of **(256, 256, 3)**,
  corresponding to RGB images resized to 256x256 pixels.
* **Conv2D Layers**: Multiple convolutional layers are used to automatically learn hierarchical spatial features from the input images.
* **MaxPooling2D Layers**: These layers are interleaved with convolutional layers to reduce the spatial dimensions of the feature maps,
  helping to manage computational complexity and provide a degree of translational invariance.
* **Flatten Layer**: Converts the final 2D feature maps into a 1D vector, preparing the data for the fully connected dense layers.
* **Dense Layers**: Fully connected layers process the high-level features extracted by the convolutional and pooling layers to perform the final classification.
* **Activation Functions**:
    * **ReLU (Rectified Linear Unit)**: Used in the convolutional and intermediate dense layers to introduce non-linearity, enabling the model to learn complex patterns.
    * **Sigmoid**: Used in the output layer. The sigmoid function squashes the output to a value between 0 and 1, representing the probability that the input image belongs to the positive class (AI-generated).
* **Loss Function**: **Binary Crossentropy** is used as the loss function, which is standard for binary classification problems. It measures the difference between the predicted probabilities and the true labels.
* **Optimizer**: The **Adam optimizer** is used to update the model's weights during training, known for its efficiency and effectiveness.

### Dataset

The project utilizes a dataset specifically collected for this classification task.

* **Real Images**: Collected from publicly available facial datasets representing diverse demographics.
* **AI-Generated Images**: Sourced from GAN-based generators producing highly realistic synthetic faces.

💡 You can also create your own dataset by collecting images manually. One way is to use the "**Download All Images**" extension in your browser to scrape images from Google’s image search.

### Results

Here’s an example output from this model:

![Example Result](https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot_example_result%20(1).jpeg)

The trained model demonstrates **high accuracy** in distinguishing between real and AI-generated face images on the dataset it was trained on.
Although, to ensure **generalization**, testing on unseen data and applying **regularization techniques** like dropout and data augmentation are recommended.

---

## 🛠️ Setup and Installation

To set up the projects locally:

1.  **Install Dependencies**: Ensure you have Python installed and install the required libraries (TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn, OpenCV, potentially `python-magic`).

2.  **Prepare Datasets**:
    * Obtain the RVF10K dataset and place it where accessible, updating the relevant path in `Deepfake-Face-Detection-with-Xception.ipynb`.
    * Obtain the dataset for the custom CNN or you can make it on your own and place it in an `imagedata` directory or update the relevant path in `Real vs. AI-Generated Face Classification model.ipynb`.

3.  **Run Notebooks**: Open and execute the cells in the respective `.ipynb` notebooks (`Deepfake-Face-Detection-with-Xception.ipynb` and `Real vs. AI-Generated Face Classification model.ipynb`).

## ✨ Future Improvements

* **Dataset Expansion:** Train both models on larger and more diverse datasets.
* **Hyperparameter Tuning:** Optimize hyperparameters for both model architectures.
* **Explore Other Architectures:** Investigate other cutting-edge CNNs, Vision Transformers, or specialized deepfake detection methods.
* **Anomaly Detection Techniques:** Explore treating deepfake detection as an anomaly detection problem.
* **Deployment:** Develop a user-friendly interface for prediction.

Feel free to contribute to this project by opening issues or submitting pull requests!!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
© 2025 indranil143
