# DAY 159 - Fine-Tuning GPT-2: The "Hello World" of LLM Training

### The Goal
Today, I explored the fundamentals of fine-tuning large language models by working with GPT-2. The goal was to understand how to adapt a pre-trained model to generate text in a specific style - in this case, inspirational quotes.

---

### The Concept: Causal Language Modeling (CLM)
GPT models are trained on one simple but powerful task: **Predict the next token**.

Given a sequence of tokens, the model learns to predict what comes next:
P(w_t | w_1, w_2, ..., w_{t-1})

Where:
- `w_t` is the token we want to predict
- `w_1, w_2, ..., w_{t-1}` are all the previous tokens in the sequence

This is called **Causal Language Modeling** because the model can only look at past tokens (left-to-right), not future ones.

**Example:**
- Input: "The quick brown fox"
- Model predicts: "jumps"
- Next input: "The quick brown fox jumps"
- Model predicts: "over"

The model learns by seeing millions of these examples during training, gradually understanding patterns in language.
---

### Why GPT-2 in 2025?
While larger models like Llama 3 and Mistral are state-of-the-art, GPT-2 remains the perfect learning tool because:

1. **Fast**: Trains in minutes on a free Colab T4 GPU
2. **Transparent**: Small enough to understand every layer and attention head
3. **Foundational**: Uses the same Decoder-only Transformer architecture as modern giants
4. **Accessible**: 124M parameters vs billions in newer models

---

### The Objective
Fine-tune GPT-2 on a dataset of inspirational quotes to teach it a "philosophical" writing style.

**Expected Results:**
- Before Training: Generic, rambling text
- After Training: Quote-like, philosophical statements

---

### Step-by-Step Implementation

#### 1. Setup and Dependencies
We need three key libraries:
- `transformers`: For the GPT-2 model and tokenizer
- `datasets`: For loading and processing data
- `accelerate`: For efficient training

#### 2. Load the Dataset
We used the **Abirate/english_quotes** dataset from Hugging Face, containing:
- 2,508 English quotes
- Author names
- Tags/categories

The dataset was split 90/10 for training and validation.

#### 3. Tokenization - The Critical Step
This is where most beginners make mistakes. For Causal Language Modeling:

**Key Requirements:**
- Append the EOS (End of String) token to every example
- Set max_length to prevent memory issues
- Use padding to create uniform batch sizes

**Format:** `Quote + <|endoftext|>`

Why EOS matters:
- Tells the model when to stop generating
- Without it, the model doesn't learn proper sentence boundaries
- Results in endless rambling during generation

#### 4. Data Collator
The `DataCollatorForLanguageModeling` with `mlm=False`:
- Automatically shifts labels for CLM training
- Input: tokens [0:n-1]
- Target: tokens [1:n]
- This teaches the model to predict the next token

#### 5. Training Configuration
We used conservative settings suitable for a free Colab GPU:
- 3 epochs (complete passes through the data)
- Batch size of 8
- Learning rate: 2e-5
- Weight decay: 0.01 for regularization

---

### Results and Why They're Not Impressive

#### What We Got:
The model generates text that resembles quotes, but the quality is limited. Here's why:

**1. Dataset Size**
- Only ~2,500 training examples
- Modern LLMs are trained on billions of examples
- GPT-2 itself was pre-trained on 40GB of text

**2. Training Duration**
- 3 epochs = ~850 training steps
- Full GPT-2 pre-training: Millions of steps
- We only "nudged" the model slightly

**3. Model Capacity**
- GPT-2 (124M parameters) is tiny by today's standards
- Llama 3: 8-70 billion parameters
- GPT-4: Estimated 1+ trillion parameters

**4. Data Quality**
- Short quotes lack the complexity needed for sophisticated generation
- No reinforcement learning from human feedback (RLHF)
- No instruction tuning

#### What This Demonstrates:
This project shows the **fundamental mechanics** of LLM fine-tuning:
- How tokenization works
- How CLM training shifts labels
- How to use the Trainer API
- How generation parameters affect output

**Think of this as:**
- Learning to drive in a parking lot (GPT-2 fine-tuning)
- vs. Racing in Formula 1 (Training GPT-4 from scratch)

Both use the same principles, just different scales!

---

### Generation Parameters Explained

#### Temperature
Controls randomness in text generation:
- **Low (0.2)**: Conservative, repetitive, safe
- **Medium (0.7)**: Balanced creativity
- **High (1.5)**: Creative, chaotic, potentially nonsensical

#### Beam Search
- Explores multiple generation paths simultaneously
- `num_beams=3` means tracking 3 candidate sequences
- Produces more coherent but less diverse outputs

#### No Repeat N-gram
- `no_repeat_ngram_size=2` prevents repeating 2-word phrases
- Reduces boring repetition
- May occasionally hurt fluency

---

### Real-World Applications

This same approach scales to production use cases:

1. **Customer Service Chatbots**
   - Fine-tune on company-specific conversations
   - Learn brand voice and policies

2. **Content Generation**
   - Blog posts, marketing copy, product descriptions
   - Fine-tune on existing high-quality content

3. **Code Generation**
   - Fine-tune on company codebases
   - Learn internal patterns and conventions

4. **Medical/Legal Text**
   - Domain-specific language models
   - Requires much larger datasets and compute

---

### Sources and References

#### Libraries and Tools
- Transformers: https://huggingface.co/docs/transformers
- Datasets: https://huggingface.co/docs/datasets
- GPT-2 Model: https://huggingface.co/gpt2
- English Quotes Dataset: https://huggingface.co/datasets/Abirate/english_quotes

#### Concepts and Papers
- GPT-2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Language Models are Few-Shot Learners (GPT-3): https://arxiv.org/abs/2005.14165

#### Tutorials and Guides
- Hugging Face Fine-tuning Tutorial: https://huggingface.co/docs/transformers/training
- Understanding GPT-2: http://jalammar.github.io/illustrated-gpt2/
- Causal Language Modeling: https://huggingface.co/docs/transformers/tasks/language_modeling

---

### From GPT-2 to Modern LLMs

The code we wrote is **90% identical** to what you'd use for Llama 3 or Mistral. The main differences:

1. **Model Size**: Need LoRA/QLoRA for memory efficiency
2. **Data Scale**: Thousands vs millions of examples
3. **Training Time**: Hours/days vs minutes
4. **Compute**: Multiple GPUs vs single GPU

**What Stays the Same:**
- Tokenization approach
- EOS token handling
- Data collator logic
- Training loop structure
- Generation parameters

---

### Key Takeaways

**What This Project Teaches:**
1. How LLMs learn from data (next-token prediction)
2. Why tokenization matters (EOS tokens, padding)
3. How to use the Trainer API
4. How generation parameters affect output
5. Why scale matters in LLM performance

**Why Results Aren't Amazing:**
- Limited training data (2,500 examples)
- Short training time (3 epochs)
- Small model size (124M parameters)
- Simple task (quote generation)

**The Value:**
Understanding these fundamentals is essential before working with larger models. GPT-2 fine-tuning is like learning algebra before calculus - you need the foundation first.

---

### Next Steps

To improve results:
1. **More Data**: Use 10,000+ quotes
2. **Longer Training**: 10-20 epochs
3. **Better Generation**: Add top-k, top-p sampling
4. **Evaluation**: Use perplexity metrics
5. **Instruction Tuning**: Add prompts like "Write a quote about..."

To scale up:
1. Use LoRA for efficient fine-tuning
2. Try Llama 2/3 or Mistral models
3. Use instruction-following datasets
4. Implement RLHF for alignment