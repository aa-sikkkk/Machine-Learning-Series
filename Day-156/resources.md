# DAY 156 - Hyperparameter Tuning with Optuna

### The Goal
Today, I explored the art of **hyperparameter tuning** and how to optimize machine learning models efficiently. Instead of relying on manual tuning or brute-force methods like grid search, I used **Optuna**, a state-of-the-art library for hyperparameter optimization. The goal was to find the best configuration for a neural network on the **FashionMNIST** dataset while minimizing computational cost.

---

### The Problem
Hyperparameter tuning is often a bottleneck in machine learning workflows. Common challenges include:
1. **Learning Rate:** Too high leads to divergence, too low slows convergence.
2. **Batch Size:** Too large causes memory issues, too small introduces noise.
3. **Model Complexity:** Too many layers/neurons lead to overfitting, too few lead to underfitting.

Traditional methods like **grid search** or **random search** are inefficient:
- **Grid Search:** Explores all combinations but is computationally expensive.
- **Random Search:** Faster but lacks direction.

---

### The Solution: Bayesian Optimization with Optuna
Optuna uses **Bayesian Optimization** with a **Tree-structured Parzen Estimator (TPE)**. Instead of blindly trying combinations, it:
1. Treats hyperparameter tuning as a probability problem.
2. Learns from past trials to predict which hyperparameters are likely to improve performance.
3. Prunes bad trials early to save computational resources.

---

### The Stack
Here’s what I used for this experiment:
1. **Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), a more challenging alternative to MNIST.
2. **Model:** A simple feedforward neural network with:
   - A shared backbone (Flatten + Dense layers).
   - Dropout for regularization.
3. **Optimization Library:** [Optuna](https://optuna.org/) for hyperparameter tuning.

---

### The Experiments

#### 1. **Defining the Search Space**
Instead of hardcoding values, I defined distributions for the hyperparameters:
- **Learning Rate:** Logarithmic scale ($1e^{-5}$ to $1e^{-1}$).
- **Optimizer:** Categorical choice (Adam, SGD).
- **Number of Neurons:** Integer range (32 to 128).
- **Dropout Rate:** Continuous range (0.1 to 0.5).

#### 2. **Pruning Trials**
Optuna pruned bad trials early. For example:
- If a model performed poorly after 2 epochs, it was terminated immediately.
- This saved significant computational time.

#### 3. **Optimization Results**
After 20 trials, Optuna found the best configuration:
- **Learning Rate:** 0.0042
- **Optimizer:** Adam
- **Number of Neurons:** 90
- **Dropout Rate:** 0.22
- **Best Accuracy:** 81.6%

---

### Results and Insights

#### 1. **Optimization History**
The optimization history plot showed rapid improvement in the first few trials, with accuracy plateauing around 82%. This indicated that further tuning on these specific parameters would yield diminishing returns.

#### 2. **Parameter Importance**
Optuna's parameter importance analysis revealed:
- **Optimizer (43%):** The choice between Adam and SGD was the most critical factor.
- **Dropout (35%):** Regularization played a significant role in improving generalization.
- **Learning Rate (14%) & Neurons (8%):** Surprisingly, the number of neurons had minimal impact.

#### 3. **Strategic Insights**
If I had tuned manually, I might have wasted time tweaking the learning rate or adding more neurons. Optuna proved that the optimizer choice and dropout rate were the real game-changers.

---

### Real-World Applications
Hyperparameter tuning with Optuna can be applied to various machine learning tasks, including:
1. **Image Classification:** Optimizing CNN architectures for datasets like CIFAR-10 or ImageNet.
2. **NLP:** Fine-tuning Transformers (e.g., BERT, GPT) for text classification or summarization.
3. **Reinforcement Learning:** Tuning reward functions and exploration strategies.

---

### Sources and References

#### Libraries and Tools
- **Optuna:** [Optuna Documentation](https://optuna.readthedocs.io/) for hyperparameter optimization.
- **PyTorch:** [PyTorch](https://pytorch.org/) for building and training the neural network.
- **FashionMNIST Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) for benchmarking.

#### Concepts and Techniques
- **Bayesian Optimization:** [Bayesian Optimization](https://arxiv.org/abs/1807.02811) for understanding the underlying methodology.
- **Tree-structured Parzen Estimator (TPE):** [TPE Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) for the algorithm used by Optuna.
- **Pruning in Hyperparameter Tuning:** [Optuna Pruning](https://optuna.readthedocs.io/en/stable/tutorial/pruning.html) for early stopping of bad trials.

#### Blog Posts and Articles
- **Hyperparameter Tuning with Optuna:** [Towards Data Science](https://towardsdatascience.com/) - A practical guide to using Optuna.
- **Bayesian Optimization for ML:** [Sebastian Ruder's Blog](https://ruder.io/) - A deep dive into Bayesian Optimization.
- **Efficient Hyperparameter Tuning:** [Hugging Face Blog](https://huggingface.co/blog/) - Tips for tuning large models.

---

### Final Thoughts
This experiment demonstrated the power of **Bayesian Optimization** for hyperparameter tuning. By leveraging Optuna, I was able to:
1. Save time by pruning bad trials early.
2. Gain insights into which hyperparameters mattered most.
3. Achieve state-of-the-art performance with minimal computational cost.

Hyperparameter tuning is no longer guesswork—it's a systematic, data-driven process. If you're looking to optimize your models efficiently, I highly recommend exploring Optuna and the resources above.