# DAY 158 - Advanced Text Processing: From TF-IDF to Transformers

### The Goal
The goal of this project was to explore text representation techniques and understand how they have evolved over time. We started with traditional methods like TF-IDF, moved to Word Embeddings, and finally explored Contextual Embeddings using Transformers. By the end, we built a Semantic Search Engine capable of understanding the meaning of queries.

---

### Why Text Processing Matters
Computers cannot understand text directly—they only understand numbers. Text processing is the process of converting raw text into numerical representations that machine learning models can work with.

---

### The Evolution of Text Representations

#### 1. Bag of Words (TF-IDF)
- What it does: Counts word occurrences in a document and assigns weights based on their importance.
- Strengths: Simple and interpretable, works well for small datasets.
- Weaknesses: Ignores context, produces sparse matrices.

#### 2. Word Embeddings (Word2Vec, GloVe)
- What it does: Maps each word to a dense vector in a high-dimensional space.
- Strengths: Captures semantic relationships, words with similar meanings are close in the vector space.
- Weaknesses: Static representations, same word has the same vector regardless of context.

#### 3. Contextual Embeddings (Transformers)
- What it does: Encodes entire sentences into dense vectors, capturing the meaning of words in context.
- Strengths: State-of-the-art semantic understanding, handles polysemy.
- Weaknesses: Computationally expensive.

---

### Implementation Details

#### TF-IDF Implementation
TF-IDF highlights words that are frequent in a document but rare across the corpus.

Formula: w(i,j) = tf(i,j) * log(N / df(i))

#### Word Embeddings with GloVe
Word embeddings map words to dense vectors. Example: King - Man + Woman = Queen

#### Contextual Embeddings with SBERT
SBERT encodes entire sentences into dense vectors, capturing their meaning.

---

### Results and Observations

1. TF-IDF: Simple and interpretable but struggles with synonyms and context.
2. Word Embeddings: Captures semantic relationships but limited by static representations.
3. Contextual Embeddings: State-of-the-art performance, handles context effectively.

---

### Real-World Applications
1. Search Engines: Build semantic search engines that understand user intent.
2. Chatbots: Improve chatbot responses by understanding the context of user queries.
3. Recommendation Systems: Recommend content based on semantic similarity.

---

### Sources and References

#### Libraries and Tools
- Scikit-Learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Gensim: https://radimrehurek.com/gensim/
- Sentence Transformers: https://www.sbert.net/

#### Concepts and Techniques
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Word Embeddings: https://arxiv.org/abs/1301.3781
- Contextual Embeddings: https://arxiv.org/abs/1810.04805

#### Blog Posts and Tutorials
- TF-IDF vs. Word2Vec: https://towardsdatascience.com/
- Semantic Search with SBERT: https://www.sbert.net/examples/applications/semantic-search/README.html

---

### Final Thoughts
This project demonstrated the evolution of text representation techniques, from simple word counts to state-of-the-art contextual embeddings. By understanding these methods, you can choose the right approach for your NLP tasks.