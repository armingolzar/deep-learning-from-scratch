# CIFAR-10 Classification — Three Training Approaches

**Author:** Armin Golzar  
[LinkedIn](https://www.linkedin.com/in/armingolzar/)

---

## Overview

This project implements CIFAR-10 image classification using three different training methods:

1. **Keras `model.fit()`**
2. **Custom training loop using `tf.GradientTape`**
3. **Fully custom layers (`MyConv2D`, `MyDense`, `MyFlatten`) + custom training loop**

The focus of this project is **learning the internals of training**, not achieving maximum accuracy.  
All methods achieve approximately **71–73% accuracy** with no augmentation or generalization techniques applied.

---

## Model Architecture

The CNN model used:

```python
input_layer = Input(shape=(32, 32, 3), name="input_layer")
conv1 = Conv2D(32, (3, 3), activation="relu", name="conv1")(input_layer)
conv2 = Conv2D(32, (3, 3), activation="relu", name="conv2")(conv1)
maxpool1 = MaxPooling2D(name="maxpool1")(conv2)
conv3 = Conv2D(64, (3, 3), activation="relu", name="conv3")(maxpool1)
conv4 = Conv2D(64, (3, 3), activation="relu", name="conv4")(conv3)
flatten = Flatten(name="flatten")(conv4)
dense1 = Dense(64, activation="relu", name="dense1")(flatten)
output = Dense(10, activation="softmax", name="output")(dense1)
```

## Project Structure

```bash
project_root/
│
├── data/                     # CIFAR-10 or custom data
├── models/                   # Saved models
├── assets/                   # Plots, logs, and visual assets
├── src/
│   ├── __init__.py
│   ├── config.py             # Configurations and hyperparameters
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── inference.py          # Predict on new images
│   ├── model.py              # Functional model architecture
│   ├── train.py              # Keras model.fit() training
│   ├── train_custom_loop.py  # Custom training loop
│   ├── train_from_scratch.py # Custom layers + custom loop
│   └── utils.py              # Helper functions
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Dataset

- CIFAR-10: 50,000 training images, 10,000 test images
- 10 classes, images of size 32×32×3
- Normalized to [0, 1]
- No augmentation applied

## Training Methods

1. Keras model.fit()
- Uses built-in Keras layers and training API
- Baseline for comparison

2. Custom Training Loop
- Manual forward pass
- Loss computed manually
- Gradients computed with tf.GradientTape
- Optimizer applied manually

3. Custom Layers + Custom Training Loop
- Fully manual MyConv2D, MyDense, MyFlatten layers
- Manual weight initialization
- Forward pass logic implemented from scratch
- Training loop fully manual

## 📊 Results

Even without augmentation or regularization, all three training methods achieve similar performance:

| Method                                | Accuracy      |
|---------------------------------------|---------------|
| Keras `model.fit()`                   | ~71–73%       |
| Custom training loop                  | ~71–73%       |
| Custom layers + custom loop           | ~71–73%       |

> Note: Accuracy is not the focus of this project. The main goal is to understand the internals of training and layer implementation.

---

## ▶️ How to Run

Since the project is modular, run scripts as Python modules:

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Standard Keras training
```bash
python -m src.train
```

### 3. Custom training loop
```bash
python -m src.train_custom_loop
```

### 4. Custom layers + custom loop
```bash
python -m src.train_from_scratch
```

## Key Concepts For Learning

- Writing custom layers compatible with TensorFlow
- Understanding build() and call() methods
- Forward and backward pass mechanics
- Manual gradient computation and weight updates
- TensorFlow variable naming and scoping
- Differences between Functional API and subclassed models
- Modular project organization

## Author
**Armin Golzar** <br>
AI Specialist — Deep Learning <br> 
[LinkedIn](https://www.linkedin.com/in/armingolzar/)