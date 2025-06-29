# Deepfake Face Detection using Deep Learning 🤖👤
**(Real-or-AI-Generated-Face-Detection)**

---

# 🔄 Alternative Approach: Custom CNN Model 
An earlier project detailed in `Real vs. AI-Generated Face Classification model.ipynb` explored a custom CNN:
* **Architecture:** Input **(256, 256, 3)**, multiple Conv2D & MaxPooling2D layers, Flatten, Dense layers with ReLU, and a Sigmoid output.
* **Training:** Adam optimizer, Binary Crossentropy loss.
* **Dataset:** Custom collected real and GAN-generated faces (or,💡build your own, e.g., using browser extensions to scrape images).
* **Result Example:**
    ![Example Result](https://github.com/indranil143/Real-or-AI-Generated-Face-Detection/blob/main/SS/Screenshot_example_result%20(1).jpeg)
    *Achieved high accuracy on its specific training set; generalization benefits from techniques like dropout and data augmentation.*

---


Feel free to contribute by opening issues or submitting pull requests!

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
© 2025 indranil143
