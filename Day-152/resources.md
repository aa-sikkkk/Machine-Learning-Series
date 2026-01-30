# DAY 152 - Visualizing the "Ghost" in the Machine: BERT Attention

### The Goal
Today, I wanted to dive deeper into the inner workings of **BERT (Bidirectional Encoder Representations from Transformers)** and visualize how it processes language. Specifically, I explored **self-attention** mechanisms and how BERT assigns relevance scores between words in a sentence.

The objective was to answer questions like:
1. How does BERT understand relationships between words (e.g., adjectives modifying nouns)?
2. How does it resolve coreferences (e.g., what does "it" refer to in a sentence)?
3. Why do special tokens like `[SEP]` and `[CLS]` receive so much attention?

---

### The Concept: Self-Attention
Transformers like BERT process all words in a sentence simultaneously, rather than sequentially. To understand a word, the model "attends" to every other word in the sentence, assigning a relevance score.

The mathematical formula for self-attention is:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query ($Q$):** The word asking for context (e.g., "bank" in "river bank").
- **Key ($K$):** The word offering context (e.g., "river").
- **Value ($V$):** The actual content of the word.
- **Score:** The dot product $QK^T$ determines how much "attention" is paid.

---

### The Stack
Here’s what I used for this experiment:
1. **Model:** [BERT Base Uncased](https://huggingface.co/bert-base-uncased), the standard workhorse of NLP.
2. **Visualization Tool:** [BertViz](https://github.com/jessevig/bertviz) for interactive attention visualizations.
3. **Libraries:** [Transformers](https://github.com/huggingface/transformers) for loading the model and tokenizer.

---

### Visualizations

#### 1. **Head View**
This visualization shows the attention patterns of individual attention heads in BERT. Each layer has 12 heads, and BERT Base has 12 layers, resulting in 144 distinct attention mechanisms.

- **Test Sentence:** "The animal didn't cross the street because it was too tired."
- **Key Observations:**
  - Early layers focus on local relationships (e.g., adjacent words).
  - Mid-to-late layers focus on global relationships (e.g., resolving "it" to "animal").
  - Special tokens like `[SEP]` and `[CLS]` often act as "parking spots" for unused attention.

#### 2. **Model View**
This provides a bird's-eye view of all 144 attention heads at once. It helps identify:
- **Grid Patterns:** Heads attending to the current position (self-preservation).
- **Diagonals:** Heads attending to the previous/next word.
- **Vertical Columns:** Heads fixating on specific tokens like `[SEP]`.

#### 3. **Neuron View**
This visualization breaks down the self-attention mechanism into its components:
- **Query Vectors ($Q$):** What the word is looking for.
- **Key Vectors ($K$):** What the word offers.
- **Dot Product ($QK^T$):** The similarity score between them.

---

### Sources and References

#### Libraries and Tools
- **Transformers Library:** [Hugging Face Transformers](https://github.com/huggingface/transformers) for loading the BERT model and tokenizer.
- **BertViz:** [BertViz](https://github.com/jessevig/bertviz) for interactive attention visualizations.
- **PyTorch:** [PyTorch](https://pytorch.org/) for running the model and computations.

#### Concepts and Techniques
- **BERT (Bidirectional Encoder Representations from Transformers):** [BERT Paper](https://arxiv.org/abs/1810.04805) for understanding the architecture and training methodology.
- **Self-Attention Mechanism:** [Attention is All You Need](https://arxiv.org/abs/1706.03762) for the original Transformer paper.
- **Coreference Resolution:** [Coreference Resolution in NLP](https://web.stanford.edu/~jurafsky/slp3/21.pdf) for understanding how models resolve pronouns like "it" or "they".

#### Blog Posts and Articles
- **Visualizing Attention in Transformers:** [BertViz Blog](https://medium.com/the-artificial-impostor/visualizing-attention-in-transformers-bertviz-445e5b3e5b24) - A detailed guide on using BertViz.
- **How BERT Works:** [Jay Alammar's Blog](http://jalammar.github.io/illustrated-bert/) - A visual and intuitive explanation of BERT.
- **Understanding Self-Attention:** [Sebastian Ruder's Blog](https://ruder.io/multi-head-attention/) - A deep dive into self-attention and multi-head attention.

---

### Final Thoughts
This experiment was a fascinating way to peek inside the "black box" of BERT and understand how it processes language. Visualizing attention patterns helped me see how the model builds relationships between words and resolves ambiguities.

If you're curious about how transformers work or want to explore attention mechanisms, I highly recommend trying out BertViz. The resources above provide a great starting point for diving deeper into the world of NLP and transformers.