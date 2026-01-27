# DAY 150 - Fine-Tuning: PEFT, LoRA, and Quantization

### The Problem with Full Fine-Tuning
Fine-tuning all 250M+ parameters of a model like `flan-t5-base` is inefficient and prone to "catastrophic forgetting" (losing previously learned knowledge).

### The Solution: LoRA (Low-Rank Adaptation)
Instead of updating the pre-trained weights $W$, we freeze them and inject trainable rank decomposition matrices $A$ and $B$ into each layer.

$$W_{new} = W + \Delta W = W + BA$$

- **$r$ (Rank):** The dimension of the low-rank matrices (e.g., 8 or 16). Lower $r$ = fewer parameters.
- **Target Modules:** We specifically target the `query` and `value` projections in the attention mechanism.

### The Stack
1. **Model:** [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) (Instruction-tuned, far superior to vanilla T5).
2. **Technique:** [QLoRA](https://arxiv.org/abs/2305.14314) (4-bit quantization + LoRA) to fit a 780M parameter model on a free Colab GPU.
3. **Metric:** [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) + [BERTScore](https://github.com/Tiiiger/bert_score) (Semantic similarity).

---

### Sources and References

#### Libraries and Tools
- **Transformers Library**: [Hugging Face Transformers](https://github.com/huggingface/transformers) for model loading and fine-tuning.
- **Datasets Library**: [Hugging Face Datasets](https://github.com/huggingface/datasets) for loading and processing datasets.
- **PEFT Library**: [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) for LoRA and other fine-tuning techniques.
- **BitsAndBytes**: [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for 4-bit quantization.
- **Accelerate**: [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for efficient training on GPUs.
- **Evaluate Library**: [Evaluate](https://github.com/huggingface/evaluate) for computing metrics like ROUGE.

#### Dataset
- **SAMSum Dataset**: [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) for dialogue summarization tasks.

#### Concepts and Techniques
- **LoRA (Low-Rank Adaptation)**: [LoRA Paper](https://arxiv.org/abs/2106.09685) for parameter-efficient fine-tuning.
- **QLoRA**: [QLoRA Paper](https://arxiv.org/abs/2305.14314) for combining LoRA with 4-bit quantization.
- **ROUGE Metric**: [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) for evaluating text summarization.
- **BERTScore**: [BERTScore](https://github.com/Tiiiger/bert_score) for semantic similarity evaluation.

#### Additional Resources
- **FLAN-T5**: [FLAN-T5 Models](https://huggingface.co/models?search=flan-t5) on Hugging Face.
- **Gradient Accumulation**: [Gradient Accumulation in PyTorch](https://pytorch.org/tutorials/recipes/recipes/gradient_accumulation.html).

---

### Final Thoughts
LoRA and quantization represent a paradigm shift in large model fine-tuning, making advanced AI accessible to researchers and developers with limited computational resources.

While our educational demonstration used minimal training data, the techniques scale effectively to production workloads, enabling efficient deployment of customized language models for specialized tasks.

The combination of parameter-efficient fine-tuning and quantization opens new possibilities for democratizing large language model development, allowing individuals and small teams to create powerful, customized AI solutions.