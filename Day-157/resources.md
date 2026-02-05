# DAY 157 - Mini Deep Learning Project: Transfer Learning with ResNet18

### Project: NatureLens - Transfer Learning in Practice

---

### The Goal
The goal of this project was to train a deep learning model to classify images of **ants** and **bees** with high accuracy, using a small dataset (~120 images per class). This project demonstrates how **transfer learning** can be used to achieve state-of-the-art results with limited data and computational resources.

---

### The Challenge: "Small Data"
Deep learning models typically require large datasets (e.g., millions of images) to learn effectively. However, in many real-world scenarios, we only have access to small datasets. If we trained a Convolutional Neural Network (CNN) from scratch on such a small dataset, the model would likely **overfit** (memorize the training data) and fail to generalize to new images.

---

### The Solution: Transfer Learning
**Transfer Learning** allows us to leverage a pre-trained model (a model trained on a large dataset like ImageNet) and adapt it to our specific task. Instead of training the model from scratch, we reuse the **feature extraction layers** (the "brain") of the pre-trained model and only retrain the **output layer** (the "decision-making" part).

#### Why ResNet18?
- **ResNet18** is a lightweight and efficient deep learning model pre-trained on **ImageNet** (1.2 million images across 1,000 classes).
- It already knows how to detect general features like edges, textures, and shapes.
- We only need to fine-tune the last layer to classify **ants** and **bees**.

---

### Architecture Overview
1. **Input:** 224x224 RGB images.
2. **Backbone:** ResNet18 (pre-trained on ImageNet).
3. **Head:** A fully connected layer with 512 inputs and 2 outputs (for ants and bees).

---

### Step-by-Step Implementation

#### 1. **Dataset: Hymenoptera (Ants vs. Bees)**
We used the **Hymenoptera dataset**, which contains:
- **Training Set:** ~120 images per class.
- **Validation Set:** ~75 images per class.

The dataset was downloaded from the PyTorch tutorial repository.

#### 2. **Data Augmentation**
To combat overfitting, we applied **data augmentation** to artificially expand the dataset:
- **RandomResizedCrop:** Randomly crops and resizes the image to 224x224.
- **RandomHorizontalFlip:** Flips the image horizontally with a 50% chance.
- **Normalization:** Scales pixel values to have a mean of [0.485, 0.456, 0.406] and a standard deviation of [0.229, 0.224, 0.225] (ImageNet normalization).

#### 3. **Pre-trained ResNet18**
We loaded the pre-trained ResNet18 model and modified the final fully connected layer:
- Original: `Linear(in_features=512, out_features=1000)`
- Modified: `Linear(in_features=512, out_features=2)`

We froze the convolutional layers (optional) to prevent them from being updated during training, focusing only on retraining the final layer.

#### 4. **Training the Model**
We trained the model for 5 epochs using:
- **Loss Function:** CrossEntropyLoss (for multi-class classification).
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum.
- **Learning Rate Scheduler:** Decays the learning rate by a factor of 0.1 every 7 epochs.

#### 5. **Visualizing Predictions**
We created a function to visualize the model's predictions on the validation set. This helps us understand how well the model performs on unseen data.

---

### Results and Observations

#### 1. **Training Results**
- The model achieved **>90% accuracy** on the validation set after just 2 minutes of training.
- The training and validation loss decreased steadily, indicating that the model was learning effectively.

#### 2. **Visualization**
The model correctly classified most images in the validation set, demonstrating its ability to generalize to unseen data.

#### 3. **Key Insight**
We didn't teach the model to "see." Instead, we taught a model that already knew how to "see" (ResNet18) to recognize ants and bees.

---

### Real-World Applications
This pipeline can be adapted for various real-world tasks, such as:
1. **Manufacturing:** Detecting defects in products (e.g., normal vs. cracked).
2. **Medical Imaging:** Classifying X-rays (e.g., healthy vs. pneumonia).
3. **Security:** Face recognition (e.g., authorized vs. unauthorized).

---

### Sources and References

#### Libraries and Tools
- **PyTorch:** [PyTorch](https://pytorch.org/) for building and training the model.
- **Torchvision:** [Torchvision](https://pytorch.org/vision/stable/index.html) for pre-trained models and data augmentation.
- **Matplotlib:** [Matplotlib](https://matplotlib.org/) for visualizing predictions.

#### Dataset
- **Hymenoptera Dataset:** [PyTorch Hymenoptera Dataset](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

#### Concepts and Techniques
- **Transfer Learning:** [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/) for understanding the concept.
- **ResNet18:** [ResNet Paper](https://arxiv.org/abs/1512.03385) for the original ResNet architecture.
- **Data Augmentation:** [Data Augmentation in PyTorch](https://pytorch.org/vision/stable/transforms.html).

#### Blog Posts and Tutorials
- **Transfer Learning with PyTorch:** [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) - A step-by-step guide.
- **Understanding ResNet:** [Towards Data Science](https://towardsdatascience.com/) - A beginner-friendly explanation of ResNet.
- **Data Augmentation Techniques:** [Analytics Vidhya](https://www.analyticsvidhya.com/) - Practical tips for augmenting small datasets.

---

### Final Thoughts
This project demonstrated the power of **transfer learning** for solving real-world problems with limited data. By leveraging a pre-trained model like ResNet18, we achieved high accuracy with minimal training time and computational resources.

If you're new to deep learning, transfer learning is a great way to get started. The resources above provide a solid foundation for implementing similar projects in your own domain.