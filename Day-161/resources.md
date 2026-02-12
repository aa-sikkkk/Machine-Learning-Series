# DAY 161 - Building a Mini-Google: Extractive RAG Pipeline

### The Goal
Today, I combined two previous projects into a single intelligent system called **Extractive RAG (Retrieval-Augmented Generation)**. 
The idea was to build a system that can answer questions by first finding the right document and then extracting the precise answer from it. 
Think of it as building your own "Mini-Google."

---

### How Extractive RAG Works

Imagine you ask: "Who walked on the moon?"

The system works in two stages:

**Stage 1 - The Retriever (The Librarian)**
- Uses SBERT to convert your query and all documents into vectors
- Finds the document with the highest cosine similarity (semantic match)
- Goal: Narrow down the haystack to find the one needle

**Stage 2 - The Reader (The Analyst)**
- Uses DistilBERT-QA to read the retrieved document
- Extracts the exact answer span from that document
- Goal: Give you the specific fact, not the whole page

**Example Flow:**
- Query: "Who walked on the moon?"
- Retriever scans 1,000 documents and finds the paragraph about Apollo 11
- Reader reads that paragraph and extracts "Neil Armstrong"

---

### The Stack

Here is what I used for this project:

1. **Retriever Model:** all-MiniLM-L6-v2 (Sentence-BERT) for semantic search
2. **Reader Model:** distilbert-base-cased-distilled-squad for extractive QA
3. **Knowledge Base:** 5 documents covering Space, Biology, History, Technology, and Geography

---

### Step-by-Step Implementation

#### 1. The Knowledge Base
In a real application, this would be a database, a collection of PDFs, or a company wiki. For this demo, I simulated a library with 5 distinct topics:

- **Space:** Apollo 11 mission and the moon landing
- **Biology:** Mitochondria and cell biology
- **History:** The Great Wall of China
- **Technology:** Python programming language
- **Geography:** The Amazon River

#### 2. The Retriever
The retriever uses SBERT (all-MiniLM-L6-v2) to:
- Convert all documents into dense vector embeddings
- Convert the user query into a vector
- Compute cosine similarity between the query and all documents
- Return the most relevant document

#### 3. The Reader
The reader uses DistilBERT fine-tuned on SQuAD to:
- Accept the query and the retrieved document as input
- Predict the start and end positions of the answer
- Extract the text span between those positions
- Return the answer along with a confidence score

#### 4. The Full Pipeline
The pipeline connects the retriever and reader:
- User asks a question
- Retriever finds the most relevant document
- Reader extracts the precise answer from that document
- System returns the answer with a confidence score

---

### Results

#### Test Case 1: Space
- Query: "Who landed on the moon?"
- Retrieved Doc Score: 0.67
- Answer: "Commander Neil Armstrong and lunar module pilot Buzz Aldrin"
- Confidence: 0.40

#### Test Case 2: Biology
- Query: "What is the powerhouse of the cell?"
- Retrieved Doc Score: 0.66
- Answer: "Mitochondria"
- Confidence: 1.00

#### Test Case 3: Technology
- Query: "Which language uses indentation?"
- Retrieved Doc Score: 0.41
- Answer: "Python"
- Confidence: 0.93

#### Observations
- The retriever successfully identified the correct document every time
- Confidence scores vary based on how explicitly the answer appears in the text
- The biology question scored 1.00 confidence because the answer is stated almost verbatim
- The space question scored lower because the answer requires combining information from the passage

---

### The Strategic Win

We built a fully functional **Open-Domain Question Answering System** with three key advantages:

1. **Scalability:** We can add 1,000,000 more documents to the knowledge base. The retriever will still find the right one in milliseconds using vector similarity.

2. **Precision:** The reader does not hallucinate. It only extracts answers that physically exist in the retrieved text. Unlike generative models, it cannot make things up.

3. **Efficiency:** Instead of feeding a 500-page book into ChatGPT (which is expensive and slow), we only feed the single relevant paragraph to the reader.

---

### Real-World Applications

This exact architecture powers:

1. **Corporate Search Engines**
   - "How do I file expenses?"
   - Searches company policy documents and extracts the exact procedure

2. **Legal Discovery Tools**
   - "Find cases about negligence"
   - Searches thousands of legal documents and extracts relevant precedents

3. **Medical Assistants**
   - "What are the side effects of Aspirin?"
   - Searches drug databases and extracts specific information

4. **Customer Support**
   - "How do I reset my password?"
   - Searches knowledge base and returns step-by-step instructions

5. **Education**
   - "What caused World War I?"
   - Searches textbook content and extracts the relevant answer
   
---

### Sources and References

#### Libraries and Tools
- Sentence Transformers (SBERT): https://www.sbert.net/
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- all-MiniLM-L6-v2 Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- DistilBERT SQuAD Model: https://huggingface.co/distilbert-base-cased-distilled-squad
- PyTorch: https://pytorch.org/

#### Datasets and Papers
- SQuAD Dataset: https://rajpurkar.github.io/SQuAD-explorer/
- SQuAD Paper: https://arxiv.org/abs/1606.05250
- Sentence-BERT Paper: https://arxiv.org/abs/1908.10084
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- RAG Paper: https://arxiv.org/abs/2005.11401

#### Blog Posts and Tutorials
- Building RAG Systems: https://www.pinecone.io/learn/retrieval-augmented-generation/
- Semantic Search with SBERT: https://www.sbert.net/examples/applications/semantic-search/README.html
- Hugging Face QA Tutorial: https://huggingface.co/docs/transformers/tasks/question_answering
- What is RAG: https://aws.amazon.com/what-is/retrieval-augmented-generation/

---

### Key Takeaways

1. **Retrieval + Reading = RAG:** By splitting the problem into two stages, we get both speed and accuracy.
2. **No Hallucination:** Unlike generative models, extractive QA only returns text that exists in the source document.
3. **Scalable:** The retriever uses vector similarity, which scales to millions of documents efficiently.
4. **Modular:** You can swap the retriever or reader independently. Want better retrieval? Upgrade the SBERT model. Want better reading? Use a larger QA model.
5. **Cost-Effective:** Only the relevant paragraph is processed by the reader, saving compute and API costs.

---

### Final Thoughts
This project brought together semantic search and extractive QA into a unified system. The result is a lightweight but powerful question-answering pipeline that can be deployed in real-world applications.

The beauty of this architecture is its modularity. Each component can be upgraded independently, and the system can scale from 5 documents to millions without changing the core logic.

If you are building any kind of knowledge retrieval system, this RAG architecture is a solid foundation to start with.