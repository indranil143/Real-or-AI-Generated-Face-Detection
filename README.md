# Deepfake Face Detection using Deep Learning 🤖👤
**(Real-or-AI-Generated-Face-Detection)**

[![Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Concept-Deep%20Learning-purple?style=flat-square)](https://en.wikipedia.org/wiki/Deep_learning)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-red?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/Data%20Handling-NumPy-informational?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Plotting-Matplotlib-lightgrey?style=flat-square&logo=matplotlib)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-blueviolet?style=flat-square&logo=seaborn)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML%20Tools-Scikit--learn-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/stable/)
[![OpenCV](https://img.shields.io/badge/Image%20Processing-OpenCV-darkgreen?style=flat-square&logo=opencv)](https://opencv.org/)
[![Model: Xception](https://img.shields.io/badge/Model-Xception-orange?style=flat-square)](https://keras.io/api/applications/xception/)
[![Evaluation: K-Fold CV](https://img.shields.io/badge/Evaluation-K--Fold%20CV-yellowgreen?style=flat-square)](https://scikit-learn.org/stable/modules/cross_validation.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?style=flat-square&logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
AI-generated images are now almost everywhere, making it crucial to detect them before misinformation spreads or you get catfished!! 🎭  
There is a critical need to reliably detect AI-generated facial images (deepfakes) to mitigate these risks. This repository showcases a robust primary approach utilizing the **Xception network with transfer learning and K-Fold cross-validation**, alongside an alternative approach using custom Convolutional Neural Network (CNN).

## 👤 Problem Statement
The increasing realism of GAN-generated faces makes it hard to distinguish them from real ones, posing significant threats such as the proliferation of disinformation and potential reputational damage.
Therefore, we have developed advanced deep learning models specifically designed to identify AI-generated faces!

## 🚀 Main Approach: Xception-based Transfer Learning with K-Fold CV
The core solution, detailed in `Deepfake-Face-Detection-with-Xception.ipynb`, employs:

* **Data Handling:** Images from 'real' and 'fake' directories are resized to **(224, 224)** pixels, normalized, and augmented (flips, brightness, contrast, etc.).
* **Model Architecture:** **Xception** (ImageNet pre-trained) base with a custom head: Global Average Pooling, Dense layers (**L2 regularization: 0.0001, Dropout: 0.3**), Batch Normalization, and a `sigmoid` output.
* **Two-Phase Training (Adam optimizer):**
    1.  Train custom head (Xception frozen): **20 epochs**, initial Learning Rate **0.001**. Callbacks: Early Stopping (patience 10), ReduceLROnPlateau (patience 4).
    2.  Fine-tune full model (Xception unfrozen): **30 epochs**, lower Learning Rate **1e-5**. Callbacks: Early Stopping (patience 15), ReduceLROnPlateau (patience 7).
* **Cross-Validation:** **5-Fold Cross-Validation (SEED = 42)** for robust performance estimation and model checkpointing per fold.
* **Evaluation:** Metrics include Accuracy, Precision, Recall, F1-Score, AUC. Generates Classification Reports, Confusion Matrices (from pooled predictions), Training History Plots, and ROC Curves.
* **Prediction:** Loads saved `.keras` models for new image/directory predictions using a **0.5 probability threshold**.

## 🗂️ Dataset (Main Approach)
* **Name:** RVF10K
* **Source:** Kaggle - `sachchitkunichetty/rvf10k`
* **Structure:** Expected at `/kaggle/input/rvf10k/rvf10k/train` with `real` and `fake` subdirectories.
* **Size:** Approximately 7000 images.

**Some example images -** ![some image examples](https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20194643.png)

## ✅ Results (Main Approach)
Based on 5-Fold Cross-Validation on the RVF10K dataset:
<div align="center">

**K-Fold Cross-Validation Summary:**

| Metric           | Value           |
|------------------|-----------------|
| Average Accuracy | 0.7840 ± 0.0102 |
| Average AUC      | 0.8609 ± 0.0106 |

**Overall Metrics (Pooled Predictions Across All Folds):**

| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0.0 (Real)   | 0.79      | 0.77   | 0.78     | 3500    |
| 1.0 (Fake)   | 0.77      | 0.80   | 0.79     | 3500    |
| **Accuracy** |           |        | **0.78** | 7000    |
| Macro Avg    | 0.78      | 0.78   | 0.78     | 7000    |
| Weighted Avg | 0.78      | 0.78   | 0.78     | 7000    |
</div>

* **Overall ROC Curve:**
    <p align="center">
    <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot%202025-05-14%20194855.png" alt="Overall ROC Curve" height="300" width="500"/>
    </p>

**Prediction Examples:**

* **Example 1:** <p align="center"> <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Example%20(1).png" alt="Example 1 Prediction" width="600"/></p>
    * **Filename:** `LJE2ZPJRWRX.jpg`
    * **Prediction:** Fake
    * **Probability (Fake):** 0.9294
    * *Comment:* A correct prediction with high confidence that the image is Fake.

* **Example 2:** <p align="center"> <img src="https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Example%20(2).png" alt="Example 2 Prediction" width="601"/> </p>
    * **Filename:** `46915.jpg`
    * **Prediction:** Real
    * **Probability (Fake):** 0.0561
    * *Comment:* A correct prediction with high confidence that the image is Real.

---


## 🛠️ Setup and Installation
1.  **Dependencies:** Install Python (3.10+), TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn, OpenCV.
2.  **Datasets:**
    * Obtain RVF10K and update the path in `Deepfake-Face-Detection-with-Xception.ipynb`.
    * For the custom CNN, prepare your dataset (e.g., in an `imagedata` directory) or use your own and update the path in `Real vs. AI-Generated Face Classification model.ipynb`.
3.  **Run Notebooks:** Open and execute the cells in the respective `.ipynb` files.

## ✨ Future Improvements
* **Dataset Expansion:** Train on larger, more diverse datasets.
* **Hyperparameter Tuning:** Optimize for both model architectures.
* **Explore Other Architectures:** Investigate Vision Transformers (ViTs), other advanced CNNs, or specialized deepfake detection methods.
* **Anomaly Detection:** Frame deepfake detection as an anomaly problem.
* **Deployment:** Develop a user-friendly interface for prediction.

Feel free to contribute by opening issues or submitting pull requests!

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
© 2025 indranil143
