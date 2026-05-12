<div align="center">

<br/>

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██╗     ███████╗███╗   ██╗███████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██║     ██╔════╝████╗  ██║██╔════╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║     █████╗  ██╔██╗ ██║███████╗
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║     ██╔══╝  ██║╚██╗██║╚════██║
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████╗███████╗██║ ╚████║███████║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### _See the world through vectors._

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00BF63?style=for-the-badge&logo=pinecone&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas_Search-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **Multi-modal vector search engine — image similarity, zero-shot KNN classification, and text-to-image retrieval powered by CLIP embeddings.**

<br/>

---

</div>

## ✦ What is NeurōLens?

NeurōLens is a **multi-modal vector search system** that transforms raw images and natural language into high-dimensional embeddings using OpenAI's **CLIP model** — then stores, indexes, and retrieves them via **Pinecone** and **MongoDB Atlas Vector Search**.

No supervised training. No labeled pipelines. Just embeddings and geometry.

```
You give it: "wolf in snow"
It returns:  🐺🌨️ — the closest matching images in the dataset
```

<br/>

---

## ✦ Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeurōLens Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📁 Animal Dataset                                             │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐    CLIP Model (ViT-B/32)                      │
│   │   Image     │ ──────────────────────────► [512-d vector]    │
│   │   Encoder   │                                    │          │
│   └─────────────┘                                    │          │
│                                                      ▼          │
│   ┌─────────────┐                         ┌──────────────────┐  │
│   │   Text      │ ──► [512-d vector] ────►│  Vector Database │  │
│   │   Encoder   │                         │                  │  │
│   └─────────────┘                         │  ┌────────────┐  │  │
│                                           │  │  Pinecone  │  │  │
│   ┌──────────────────────────────────┐    │  └────────────┘  │  │
│   │           Query                  │    │  ┌────────────┐  │  │
│   │  "cute puppy" / query_image.jpg  │    │  │  MongoDB   │  │  │
│   └──────────────────────────────────┘    │  └────────────┘  │  │
│             │                             └──────────────────┘  │
│             ▼                                      │            │
│        Cosine                                      ▼            │
│        Similarity ◄──────────────── Top-K Results (k=5)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

## ✦ Tasks

<details>
<summary><b>🔍 Task 1 — Image Similarity Search</b></summary>

<br/>

**Goal:** Given an image, retrieve the most visually similar images from the dataset.

**How it works:**
- Every image is passed through CLIP's vision encoder → produces a `512-d` float vector
- All vectors are stored in Pinecone / MongoDB Atlas
- A query image is encoded → cosine similarity search is performed
- Top-K nearest neighbors are returned and displayed

**Expected Output:**

```
Query Image: dog_01.jpg
─────────────────────────────────────────────
Top 5 Similar Images:
  [1] dog_07.jpg     similarity: 0.97
  [2] dog_03.jpg     similarity: 0.95
  [3] dog_11.jpg     similarity: 0.93
  [4] wolf_02.jpg    similarity: 0.81
  [5] cat_04.jpg     similarity: 0.74
```

</details>

<details>
<summary><b>🏷️ Task 2 — Zero-Shot Classification via KNN</b></summary>

<br/>

**Goal:** Classify an unknown image without training a classifier.

**How it works:**
- Each stored vector has an associated label (`cat`, `dog`, `wolf`, etc.)
- For a new query image, retrieve top-K nearest neighbors
- Apply majority voting over neighbor labels → predicted class

**Expected Output:**

```
Input Image: unknown_animal.jpg

Nearest Neighbors:
  wolf_04.jpg  → wolf   (dist: 0.12)
  wolf_09.jpg  → wolf   (dist: 0.14)
  dog_02.jpg   → dog    (dist: 0.21)
  wolf_01.jpg  → wolf   (dist: 0.23)
  wolf_07.jpg  → wolf   (dist: 0.25)

Votes: wolf=4, dog=1
Predicted Label: 🐺 wolf
```

</details>

<details>
<summary><b>💬 Task 3 — Cross-Modal Search (Text → Image)</b></summary>

<br/>

**Goal:** Search images using plain English text queries.

**How it works:**
- CLIP's text encoder converts a string query into a `512-d` vector
- This lives in the **same embedding space** as image vectors
- Cosine similarity retrieves the best image matches for the text

**Example Queries:**
```
"wolf in snow"    → returns snowy wolf images
"cute puppy"      → returns close-up dog images  
"black cat"       → returns dark-colored cat images
```

**This is why CLIP is powerful** — the model was trained on 400M image-text pairs, so text and image semantics are aligned.

</details>

<br/>

---

## ✦ Tech Stack

| Component | Tool | Role |
|---|---|---|
| Embedding Model | `openai/clip-vit-base-patch32` | Image + Text → Vector |
| Vector DB (Primary) | Pinecone | Hosted vector search, fast ANN |
| Vector DB (Comparison) | MongoDB Atlas | Vector search via `$vectorSearch` |
| Runtime | Python 3.10+, Google Colab | Development environment |
| Visualization | Matplotlib | Display query + result images |

<br/>

---

## ✦ Setup

### 1. Clone the repo

```bash
git clone https://github.com/moanaXds/neurolens.git
cd neurolens
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers pinecone-client pymongo pillow matplotlib
```

### 3. Configure credentials

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_environment
MONGODB_URI=your_atlas_connection_string
```

### 4. Add the dataset

```
neurolens/
├── dataset/
│   ├── cats/
│   ├── dogs/
│   └── wolves/
```

### 5. Run

```bash
# Task 1: Image Similarity
jupyter notebook notebooks/task1_similarity.ipynb

# Task 2: KNN Classification
jupyter notebook notebooks/task2_classification.ipynb

# Task 3: Cross-Modal Search
jupyter notebook notebooks/task3_crossmodal.ipynb
```

<br/>

---

## ✦ Project Structure

```
neurolens/
│
├── notebooks/
│   ├── task1_similarity.ipynb       # Image similarity search
│   ├── task2_classification.ipynb   # KNN classification
│   └── task3_crossmodal.ipynb       # Text → Image search
│
├── src/
│   ├── embeddings.py                # CLIP encoding logic
│   ├── pinecone_store.py            # Pinecone upsert + query
│   ├── mongodb_store.py             # Atlas vector search
│   └── visualize.py                 # Result display utilities
│
├── dataset/                         # Animal images (local)
├── report/
│   └── lab14_report.pdf             # Submission report
│
├── .env.example
├── requirements.txt
└── README.md
```

<br/>

---

## ✦ Key Concepts

**Why CLIP?**
CLIP (Contrastive Language–Image Pretraining) was trained to align image and text representations. This means `encode("a dog")` and `encode(dog_image.jpg)` produce geometrically close vectors — enabling cross-modal retrieval with zero task-specific training.

**Why Vector Databases?**
Traditional SQL `WHERE` clauses can't find "similar" things. Vector DBs use Approximate Nearest Neighbor (ANN) algorithms — like HNSW or IVFFlat — to find the closest vectors in milliseconds, even across millions of entries.

**Cosine Similarity:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
Range: `[-1, 1]` → closer to `1` = more similar.

<br/>

---

## ✦ Pinecone vs MongoDB Atlas — Comparison

| Feature | Pinecone | MongoDB Atlas |
|---|---|---|
| Type | Purpose-built vector DB | Multi-model DB + vector search |
| Setup | Fully managed, API-first | Atlas cluster + index config |
| Query Speed | Extremely fast (dedicated ANN) | Fast, but secondary to doc ops |
| Metadata Filtering | ✅ Native | ✅ via `$match` pipeline |
| Best For | Pure vector workloads | Apps needing vectors + documents |

<br/>

---

## ✦ Observations

- CLIP embeddings are **surprisingly robust** — text queries like `"wolf in snow"` retrieve contextually correct images without any fine-tuning
- KNN classification via vector search **matches simple CNN accuracy** on clean datasets with no training overhead
- Pinecone offers faster cold-start indexing; MongoDB is preferable when your application already stores structured data
- Cross-modal search degrades on **ambiguous text** — `"animal"` returns scattered results across all classes

<br/>

---

<div align="center">

---

[![GitHub](https://img.shields.io/badge/GitHub-moanaXds-181717?style=flat-square&logo=github)](https://github.com/moanaXds)

<br/>

*vectors don't lie.*

</div>
