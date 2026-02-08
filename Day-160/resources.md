
# DAY 160 - Question Answering: Teaching Machines to Read

  

### The Goal

Today, I explored **Extractive Question Answering (QA)**, a fundamental NLP task where a model reads a passage of text and extracts the precise answer to a question. This is the technology behind Google's featured snippets and virtual assistants that can answer factual questions.

  

---

  

### The Concept: Extractive QA

  

In Extractive QA, the model receives two inputs:

1.  **Context:** A paragraph of text (e.g., a Wikipedia article or document)

2.  **Question:** A query related to the text

  

**Important:** The model does NOT generate an answer from scratch. Instead, it predicts two numbers:

-  **Start Index:** The token position where the answer begins

-  **End Index:** The token position where the answer ends

  

The model then extracts the text span between these positions as the answer.

  

**Example:**

Context: "The Amazon rainforest covers 5,500,000 square kilometers." Question: "How large is the Amazon rainforest?" Model Output: Start=5, End=8 Answer: "5,500,000 square kilometers"

  
  

---

  

### Why Extractive QA Matters

  

**Real-World Applications:**

1.  **Search Engines:** Google uses this to highlight answers in search results

2.  **Legal Tech:** "What is the termination clause?" (Context: 50-page contract)

3.  **Customer Support:** "How do I reset my password?" (Context: Knowledge base)

4.  **Medical Research:** Finding specific information in research papers

5.  **Education:** Automated tutoring systems that answer questions about textbooks

  

---

  

### The Model: DistilBERT-SQuAD

  

We used **DistilBERT** fine-tuned on the **SQuAD** (Stanford Question Answering Dataset):

-  **DistilBERT:** A smaller, faster version of BERT (40% smaller, 60% faster)

-  **SQuAD:** 100,000+ question-answer pairs from Wikipedia articles

-  **Performance:** 90%+ accuracy on extractive QA tasks

  

---

  

### Step-by-Step Implementation

  

#### 1. Understanding the Raw Mechanics

  

Before using high-level APIs, let's understand what happens under the hood:

  

**The Process:**

1.  **Tokenization:** Convert "Question + [SEP] + Context" into token IDs

2.  **Forward Pass:** Model outputs start_logits and end_logits for every token

3.  **Argmax:** Find tokens with highest start and end scores

4.  **Decode:** Convert token IDs back to text

  

**What are Logits?**

Logits are raw scores (before softmax) that indicate how likely each token is to be the start or end of the answer. Higher logit = more confident the model is that this token is part of the answer.

  

**Code Example:**

```python

def  manual_qa(question, context):

# Tokenize inputs

inputs = tokenizer(question, context, return_tensors="pt")

# Get model predictions

with torch.no_grad():

outputs = model(**inputs)

# Find best start and end positions

answer_start_index = torch.argmax(outputs.start_logits)

answer_end_index = torch.argmax(outputs.end_logits) + 1

# Extract answer tokens

answer_tokens = inputs.input_ids[0, answer_start_index:answer_end_index]

answer = tokenizer.decode(answer_tokens)

return answer

```

#### 2. Using the Production Pipeline

The Hugging Face pipeline handles complex edge cases:

  

Long contexts that exceed max sequence length

Confidence scores for answers

Multiple answer candidates

Cases where no answer exists in the context

  

***Code Example***:

```python

from transformers import pipeline

  

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

  

result = qa_pipeline(question="What is the key innovation?", context=technical_context)

print(f"Answer: {result['answer']}")

print(f"Confidence: {result['score']:.4f}")

  

```

  

### How It Works: The Mathematics

Step 1: Input Encoding The model receives concatenated input:

  

    [CLS] Question tokens [SEP] Context tokens [SEP]

  

Step 2: Computing Logits For each token position i, the model computes:
  

    start_logit[i]: How likely token i is the answer start
    
    end_logit[i]: How likely token i is the answer end

Step 3: Converting to Probabilities Apply softmax to get probabilities:
 

    start_prob[i] = softmax(start_logit[i])
    
    end_prob[i] = softmax(end_logit[i])



Step 4: Answer Extraction

    answer_start = argmax(start_prob)
    
    answer_end = argmax(end_prob)
    
    answer = tokens[answer_start:answer_end+1]



### What the Visualization Shows: 

> Start Probability Distribution: Bars showing probability for each
> token being the answer start
> 
> End Probability Distribution: Bars showing probability for each token
> being the answer end
> 
> Predicted Span: The region between start and end predictions
> highlighted in green

### Key Observations:
- The model assigns very low probabilities to special tokens like [CLS] and [SEP]

- Probabilities spike dramatically at the correct answer location

- Start and end predictions form a coherent span in the text

### Example Output:

Question: **"What is the key innovation?"**

Context: "...The key innovation is the ability to process the entire input sequence in parallel..."

    Start Probabilities: Spike at token 45 ("ability")
    
    End Probabilities: Spike at token 54 ("parallel")
    
    Answer: "the ability to process the entire input sequence in parallel"
    
    Confidence: 0.9234

  

### Results and Observations

Test Case 1: Amazon Rainforest

Context: Information about the Amazon rainforest

    Q: "How much of the basin is covered by rainforest?"
    
    A: "5,500,000 km2 (2,100,000 sq mi)"
    
    Confidence: 0.9567



Test Case 2: Transformers Architecture

Context: Technical explanation of Transformers

    Q: "What paper introduced Transformers?"
    
    A: "Attention Is All You Need"

Confidence: 0.9812 

Why High Confidence? 

> The answer appears explicitly in the context
> 
> Clear linguistic patterns (e.g., "introduced in the 2017 paper 'X'")
> 
> Model trained on 100,000+ similar examples from SQuAD

## Limitations and Edge Cases

  

### When Extractive QA Fails:

  

1. Answer Not in Context

Q: "What is the capital of France?"

Context: "France is a beautiful country in Europe."

Result: Model will still extract something, even if wrong

  

2. Requires Reasoning

Q: "Is the Amazon rainforest larger than Texas?"

Context: "Amazon covers 5.5M km², Texas covers 695,000 km²"

Result: Model cannot do math, only extract text

  

3. Paraphrased Answers

Q: "How fast are Transformers?"

Context: "They process sequences in parallel, significantly reducing training time"

Result: "significantly reducing training time" (partial answer, lacks specifics)

  

4. Long Contexts

BERT models have 512 token limit

Pipeline handles this by chunking, but may miss cross-chunk answers

Building a Complete RAG System

This QA system is one component of a larger architecture called Retrieval-Augmented Generation (RAG):

  

### Step 1: Retrieval (from Day 158's Semantic Search)


Use SBERT to find relevant paragraphs from a large document collection

    Input: User question
    
    Output: Top 3-5 most relevant paragraphs

### Step 2: Extraction (Today's QA System)
 
Use extractive QA on each retrieved paragraph

Input: Question + Retrieved paragraph

    Output: Extracted answer + confidence score

Step 3: Ranking

  - Rank answers by confidence scores

- Return the best answer to the user
  

### Production Architecture:

    User Question → SBERT → Top K Paragraphs → QA Model → Ranked Answers → Best Answer
 

## Comparing Extractive vs. Generative QA

### Extractive QA (Today's Project):

- Pros: Fast, factual, explainable (shows source text)

- Cons: Limited to information in context, cannot reason

### Generative QA (GPT, Llama):

- Pros: Can synthesize information, reason, handle complex queries

- Cons: May hallucinate, harder to verify, slower

### Best Practice: Use extractive QA when:

 - You need citations/sources
 - Speed is critical
 - Factual accuracy is paramount

###   Use generative QA when:  

You need synthesis of multiple sources

The answer requires reasoning

Natural language quality is more important than speed

###   Sources and References

**Libraries and Tools**

Transformers: https://huggingface.co/docs/transformers

DistilBERT Model: https://huggingface.co/distilbert-base-cased-distilled-squad

PyTorch: https://pytorch.org/

Datasets and Papers

SQuAD Dataset: https://rajpurkar.github.io/SQuAD-explorer/

SQuAD Paper: https://arxiv.org/abs/1606.05250

DistilBERT Paper: https://arxiv.org/abs/1910.01108

BERT Paper: https://arxiv.org/abs/1810.04805

Tutorials and Guides

Hugging Face QA Tutorial: https://huggingface.co/docs/transformers/tasks/question_answering

Understanding BERT QA: http://jalammar.github.io/illustrated-bert/

Building RAG Systems: https://www.pinecone.io/learn/retrieval-augmented-generation/

Key Takeaways

###  What This Project Teaches: 

 - How extractive QA works (span prediction vs. generation)

- Understanding logits and probability distributions

- The importance of tokenization in QA tasks

- How to visualize model predictions

- Real-world applications of reading comprehension AI

###  The Strategic Win: 
We built a system that can read and extract information from text. Combined with semantic search (Day 158), this forms the foundation of:
 - Enterprise search systems
 -  Customer support bots
 -    Legal document analysis
 -    Medical information retrieval
 -    Educational AI tutors

### Understanding the Confidence Score

The confidence score is computed as:

    score = start_prob[predicted_start] * end_prob[predicted_end]

  

Interpreting Scores:
    
    Above 0.9: Very confident, answer is almost certainly correct
    
    0.7-0.9: Good confidence, likely correct
    
    0.5-0.7: Moderate confidence, verify manually
    
    Below 0.5: Low confidence, answer may be wrong

Production Tip: Set a threshold (e.g., 0.7) and fall back to "I don't know" for low-confidence predictions rather than returning potentially incorrect answers.

  
  

### Key Improvements:

1.  **Detailed Explanations:** Broke down complex concepts like logits and probabilities

2.  **Real-World Context:** Showed practical applications and limitations

3.  **RAG Architecture:** Connected this to the bigger picture of retrieval systems

4.  **Visual Thinking:** Explained what the probability distributions mean

5.  **Production Considerations:** Discussed confidence scores and thresholds

6.  **Comparison with Generative QA:** Helped readers understand when to use each approach### Key Improvements:

1.  **Detailed Explanations:** Broke down complex concepts like logits and probabilities

2.  **Real-World Context:** Showed practical applications and limitations

3.  **RAG Architecture:** Connected this to the bigger picture of retrieval systems

4.  **Visual Thinking:** Explained what the probability distributions mean

5.  **Production Considerations:** Discussed confidence scores and thresholds

6.  **Comparison with Generative QA:** Helped readers understand when to use each approach
