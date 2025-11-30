# CIFAR-10 Classification: Three Training Approaches

This project implements CIFAR-10 image classification using three different training methods, gradually moving from high-level Keras abstractions to fully manual implementations.  
The focus is on understanding training internals, not maximizing accuracy.

All three approaches achieve approximately **71–73% accuracy** without augmentation or regularization.

---

## 1. Standard Keras Training (`model.fit()`)

This version uses the built-in Keras API:

- Standard `Conv2D`, `Flatten`, and `Dense` layers  
- Training with `model.fit()`  
- Adam optimizer  
- Categorical cross-entropy loss  

This serves as the baseline model and demonstrates the simplicity of high-level Keras tools.

---

## 2. Custom Training Loop (Using Built-In Layers)

This implementation keeps Keras layers but manually handles training:

- `tf.GradientTape` for gradient calculation  
- Manual forward pass  
- Manual loss computation  
- `optimizer.apply_gradients` for updates  
- Manual metric tracking  

This provides deeper insight into how Keras executes training internally.

---

## 3. Fully Custom Layers + Custom Training Loop

In this version, core layers are implemented from scratch:

- `MyConv2D` (custom convolution layer)  
- `MyDense` (custom dense layer)  
- `MyFlatten` (custom flatten layer)  
- Manual weight creation via `add_weight`  
- Manual convolution logic  
- Fully manual training loop  

This reveals how deep learning frameworks manage:

- Weight initialization  
- Forward operations  
- Backpropagation  
- Layer naming and unique scoping  
- TensorFlow variable management  
- Custom serialization behavior  

---

## Results

All three methods produce similar results:

| Method                                | Accuracy      |
|---------------------------------------|---------------|
| Keras `model.fit()`                   | ~71–73%       |
| Custom training loop                  | ~71–73%       |
| Fully custom layers + training loop   | ~71–73%       |

The matching performance confirms that the custom layer implementations behave correctly.

---

## Architecture

The architecture is intentionally simple:

- Conv2D → ReLU  
- Conv2D → ReLU  
- Flatten  
- Dense → ReLU  
- Dense (10-class output)  

This makes it easier to debug and understand internal operations.

---

## Dataset

- CIFAR-10 dataset  
- 50,000 training images, 10,000 test images  
- 10 classes  
- Images normalized to `[0, 1]`  
- No augmentation (because the focus is training mechanics, not generalization)

---

## Usage

Run any of the following:

```bash
python train_fit.py                # Standard Keras training
python train_custom_loop.py        # Custom training loop
python train_from_scratch.py       # Custom layers + custom loop