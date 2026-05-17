#  YouTube Comment Intelligence Engine
### *An end-to-end NLP pipeline that transforms raw YouTube comment streams into structured linguistic insights, semantic clusters, and sentiment-aware comment categories.*

---

##  Project Overview

**The Problem:** YouTube comment sections are massive, unstructured, and noisy. For researchers, content creators, and community managers, manually understanding what an audience is saying — what they love, criticize, spam, or discuss — is practically impossible at scale.

**The Solution:** This project builds a complete Natural Language Processing (NLP) research system that ingests live YouTube comments via the YouTube Data API, and then progressively transforms them through a multi-stage pipeline — from raw text to meaningful, machine-labeled semantic categories. The system explores classical NLP theory (stemming, N-grams, POS tagging, HMMs) alongside modern distributional semantics (Word2Vec, GloVe, Sentence Transformers) and culminates in a trained LSTM-based comment classifier.

This project serves as a hands-on exploration of the full NLP stack — ideal as a learning reference, a research baseline, or a foundation for a production comment moderation system.

---

##  Key Features

- **Live Data Ingestion** — Fetches paginated comment threads directly from any YouTube video using the YouTube Data API v3, with automatic rate-limiting and full comment metadata (author, likes, timestamp).
- **Multi-Stage Text Preprocessing** — Applies a rigorous pipeline: URL removal, case folding, special character normalization, word tokenization, spell correction (via TextBlob), stemming (Porter Stemmer), and lemmatization (WordNet Lemmatizer).
- **Linguistic Analysis Suite** — Implements Part-of-Speech (POS) tagging, Hidden Markov Model (HMM) sequence labeling, and Named Entity Recognition (NER) using NLTK's chunking framework.
- **N-gram & Text Representation Modeling** — Generates and visualizes Word N-grams (bigrams) and Character N-grams (trigrams) via scikit-learn's `CountVectorizer`, alongside Bag-of-Words and TF-IDF feature matrices.
- **Latent Semantic Analysis (LSA)** — Applies Truncated SVD over a TF-IDF matrix to discover latent topical structure within comment text, with 2D PCA projections for visual inspection.
- **WordNet Lexical Semantics** — Explores synsets, hypernyms, hyponyms, and computes Path Similarity, Wu-Palmer Similarity, and Leacock-Chodorow Similarity between comment vocabulary.
- **Word Sense Disambiguation (WSD)** — Uses the Lesk algorithm to resolve word meaning ambiguity in context.
- **Distributional Semantics & Embeddings** — Loads pre-trained **GloVe** (50d, 100d) and **Word2Vec** models; computes and visualizes cosine similarity matrices between words; projects embedding spaces via PCA.
- **Probabilistic Topic Modeling (LDA)** — Builds a Latent Dirichlet Allocation model with Gensim to discover themes across comments; renders an interactive **pyLDAvis** visualization.
- **Semantic Clustering** — Encodes comments into dense sentence vectors using `all-MiniLM-L6-v2` (Sentence Transformers), clusters them with **K-Means**, and visualizes cluster structure with **t-SNE**.
- **Automated Comment Labeling** — Assigns human-interpretable labels (e.g., *Praise*, *Constructive Criticism*, *Spam*, *Generic*, *Unrelated*) to clusters, creating a weakly supervised training set.
- **Comment Classification** — Trains and compares a **Random Forest** classifier and a **Support Vector Machine (SVM)** on the auto-labeled embeddings.
- **LSTM Sentiment Classifier** — Builds and trains a **Bidirectional LSTM** neural network (via Keras/TensorFlow) on tokenized, padded comment sequences for multi-class comment categorization. Reports accuracy, precision, recall, F1-score, and a confusion matrix.
- **System Architecture Visualization** — Auto-generates a `graphviz` architecture diagram of the full pipeline.

---

##  Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Data Collection** | `google-api-python-client` (YouTube Data API v3) |
| **Data Handling** | `pandas`, `numpy` |
| **Classical NLP** | `nltk` (tokenization, POS, HMM, NER, WordNet, Lesk), `TextBlob` |
| **Feature Engineering** | `scikit-learn` (CountVectorizer, TfidfVectorizer, TruncatedSVD, PCA) |
| **Word Embeddings** | `gensim` (GloVe, Word2Vec via API), raw GloVe vectors (Stanford NLP) |
| **Sentence Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Topic Modeling** | `gensim` (LDA), `pyLDAvis` |
| **Clustering** | `scikit-learn` (KMeans, t-SNE) |
| **Deep Learning** | `tensorflow` / `keras` (Embedding, LSTM, Dropout, Dense) |
| **Classical ML** | `scikit-learn` (RandomForestClassifier, SVC) |
| **Visualization** | `matplotlib`, `seaborn`, `pyLDAvis`, `graphviz` |
| **Notebook Environment** | Jupyter / Google Colab |

---

##  Technical Architecture

The system is structured as a sequential, 10-stage NLP pipeline:

```
YouTube API
    │
    ▼
[1] Data Ingestion
    Paginated comment fetching via YouTube Data API v3.
    Extracts: author, comment_text, likes, published_at.
    Output: youtube_comments.csv
    │
    ▼
[2] Basic Text Preprocessing
    Sentence segmentation → Case folding → URL & special char removal
    → Word tokenization → TextBlob spell correction
    │
    ▼
[3] Advanced Text Processing
    Porter Stemming → WordNet Lemmatization
    → Token frequency analysis
    │
    ├──────────────────────┬───────────────────────┐
    ▼                      ▼                       ▼
[4] Exploratory         [5] Linguistic          [6] Semantic
    Analysis                Analysis                Analysis
    Token length dist.      N-grams (word/char)     WordNet synsets
    Word frequency          POS Tagging             Word similarity
    BoW / TF-IDF            HMM Tagging             WSD (Lesk)
                            NER                     Word2Vec / GloVe
    │                      │                       │
    └──────────────────────┴───────────────────────┘
                           │
                           ▼
              [7] Latent Semantic Analysis (LSA)
                  TF-IDF → TruncatedSVD (n=2)
                  → 2D PCA projection
                           │
                           ▼
              [8] Probabilistic Topic Modeling (LDA)
                  Dictionary → BoW corpus → LDA (k=5 topics)
                  → pyLDAvis interactive visualization
                           │
                           ▼
              [9] Semantic Clustering
                  MiniLM-v2 sentence embeddings
                  → K-Means (k=5) → t-SNE visualization
                  → Manual cluster label mapping
                           │
                           ▼
             [10] Model Training & Evaluation
                  ┌──────────────┬──────────────┐
                  ▼              ▼              ▼
            Random Forest      SVM          Bi-LSTM
            (embeddings)   (embeddings)  (token sequences)
                  └──────────────┴──────────────┘
                                 │
                                 ▼
                       Categorized Comments
                       + Evaluation Reports
                       + Confusion Matrix
```

### Key Algorithms & Mathematical Concepts

**TF-IDF Vectorization**

For a term *t* in document *d* across corpus *D*:

```
TF-IDF(t, d, D) = TF(t, d) × log(|D| / df(t))
```

Where `TF(t, d)` is the raw term count normalized by document length, and `df(t)` is the number of documents containing term *t*. This down-weights common terms and up-weights discriminative ones — forming the input matrix for LSA.

**Latent Semantic Analysis (LSA) via Truncated SVD**

Given a TF-IDF matrix `X ∈ ℝ^(m×n)` (m documents, n vocabulary terms), LSA factorizes it as:

```
X ≈ U_k × Σ_k × V_k^T
```

Where `k=2` latent components are retained. Each document is projected into a lower-dimensional *semantic latent space*, where syntactically different but semantically related comments cluster together.

**Cosine Similarity for Word Embeddings**

For two word vectors `u` and `v` in embedding space:

```
cos(u, v) = (u · v) / (||u|| × ||v||)
```

This is used to compare GloVe and Word2Vec representations, producing pairwise similarity matrices visualized as heatmaps.

**WordNet Similarity Metrics**

- **Path Similarity:** `1 / (1 + shortest_path_distance(s1, s2))` — based on the hypernym hierarchy depth.
- **Wu-Palmer (WUP) Similarity:** `2 × depth(LCS) / (depth(s1) + depth(s2))` — where LCS is the Least Common Subsumer.
- **Leacock-Chodorow (LCH):** `-log(shortest_path / (2 × max_depth))` — requires identical POS.

**LSTM Architecture**

```
Input (padded sequences, maxlen=100)
    │
    ▼
Embedding Layer (vocab=10,000, dim=64)
    │
    ▼
LSTM Layer (units=64)
    │
    ▼
Dropout (rate=0.5)      ← regularization to prevent overfitting
    │
    ▼
Dense (units=64, activation='relu')
    │
    ▼
Dense (units=num_classes, activation='softmax')
    │
    ▼
Categorical Cross-Entropy Loss + Adam Optimizer
```

---

##  Installation & Usage

### 1. Clone the Repository

```bash
git clone (https://github.com/AbhishekGitBot/My-Works-on-Natural-Language-Processing.git)
cd youtube-comment-intelligence
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install google-api-python-client pandas numpy nltk textblob \
            scikit-learn gensim sentence-transformers tensorflow \
            matplotlib seaborn pyLDAvis graphviz wordcloud
```

Download required NLTK corpora (run once):

```python
import nltk
nltk.download(['punkt', 'punkt_tab', 'wordnet', 'omw-1.4',
               'stopwords', 'averaged_perceptron_tagger_eng',
               'maxent_ne_chunker', 'maxent_ne_chunker_tab',
               'words', 'treebank', 'wordnet_ic'])
```

### 4. Configure Your YouTube API Key

Open the notebook and replace the placeholder in the Data Ingestion cell:

```python
API_KEY = "YOUR_YOUTUBE_DATA_API_V3_KEY"
```

> Get a free API key at [Google Cloud Console](https://console.cloud.google.com/) → Enable *YouTube Data API v3*.

### 5. Run the Notebook

```bash
jupyter notebook NLP_Project.ipynb
```

Or upload directly to **Google Colab** for a zero-setup GPU-accelerated environment.

### 6. Point at Any YouTube Video

```python
video_id = "YOUR_VIDEO_ID_HERE"   # e.g., "dQw4w9WgXcQ"
df = get_all_comments(video_id)
```

All downstream pipeline stages will run on the freshly collected comments.

---

##  Project Structure

```
youtube-comment-intelligence/
│
├── NLP_Project.ipynb              # Main notebook — full pipeline
├── youtube_comments.csv           # Auto-generated after data ingestion
├── glove_data/                    # GloVe vectors (downloaded at runtime)
│   └── glove.6B.50d.txt
├── youtube_comment_analysis_architecture.png   # Graphviz diagram
├── enhanced_youtube_nlp_architecture.png       # Detailed architecture
├── requirements.txt
└── README.md
```

---

##  Future Roadmap

**1. Fine-Tuning with a Domain-Specific Transformer**
Replace the generic `all-MiniLM-L6-v2` encoder with a fine-tuned `BERTweet` or `DeBERTa` model trained on social media text. This would improve embedding quality for informal language, slang, and emoji-heavy YouTube comments, yielding more coherent semantic clusters and higher downstream classifier performance.

**2. Zero-Shot & Few-Shot Comment Classification with LLMs**
Integrate an LLM-backed classification layer (e.g., via the OpenAI or Anthropic API) to enable **zero-shot labeling** of comments into custom, user-defined categories without retraining. This would eliminate the dependency on the K-Means cluster→manual-label mapping, replacing it with deterministic, prompt-driven category assignment.

**3. Real-Time Streaming Moderation Dashboard**
Extend the pipeline into a live system using the YouTube live chat API and a Streamlit dashboard. Incoming comments would be classified in near-real-time and flagged for moderation (spam, toxicity, off-topic), enabling content creators to manage community health at scale.

---


---

##  Author & Contact

**ABHISHEK SHARMA**
*AI/ML Engineer & NLP Researcher*

- 🔗 **LinkedIn:** (https://www.linkedin.com/in/abhiisheksharrma/))
- 📧 **Email:** sharrmaabhishek1@gmail.com

---

> *"Language is the latent space of human thought. This project is an attempt to map it."*
