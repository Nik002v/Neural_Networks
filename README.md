# 🧠 Neural Network from Scratch in C++

This project is a clean, modular C++ implementation of core neural network components, built from the ground up. It includes an end-to-end pipeline for training and evaluating models on an audio classification task — specifically vowel recognition — using a small dataset of audio sequences.

The primary focus is on **understanding and implementing the internals of neural networks**, not on optimizing performance. The classification task is simply a demonstration of how the framework can be used.

---

## ✅ Features

- **Modular Neural Network Framework**
  - `NeuralNetwork` class to manage training and inference
  - `Layer` class supporting various activations
  - `Loss` functions: MSE, cross-entropy
  - `Regularizer` support: L1, L2, ElasticNet
  - `Optimizer` implementations: **SGD**, **Adam**
- **Preprocessing Pipeline**
  - Train/test split
  - One-hot encoding of labels
  - Standardization of input features
- **Model Training**
  - **Denoising Autoencoder** 
  - **Feedforward Classifier** using the encoder from the autoencoder for feature extraction
- **Minimal External Dependencies**
  - Only depends on [Eigen](https://eigen.tuxfamily.org) for linear algebra

---

## 🔊 Dataset

- **500 audio examples** of vowel sounds (balanced across classes)
- Dataset available here: https://drive.google.com/file/d/1r0zLOusossMFZ0ZOQcye_iOE0Zoqrdlq/view?usp=drive_link

---

## 🏗️ Training Pipeline

1. **Preprocess the data**
   - Split into training/test sets
   - One-hot encode labels
   - Standardize features

2. **Train a denoising autoencoder**
   - Corrupt input with noise
   - Train to reconstruct clean inputs

3. **Train a classifier**
   - Use the encoder from the autoencoder as the first few layers (for feature extraction)
   - Add a classification head (fully connected layer + softmax)
   - No hyperparameter tuning (by design)

4. **Evaluate the classifier**
   - Achieves ~**70% accuracy** on the test set

---

## 📦 Dependencies

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (for matrix operations)

To install Eigen:
```bash
# On Ubuntu
sudo apt install libeigen3-dev

# Or clone manually
git clone https://gitlab.com/libeigen/eigen.git
