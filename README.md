# SetConnect Medical AI Diagnostic Workstation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://gradio.app/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/atlas)
[![AWS S3](https://img.shields.io/badge/AWS-S3-569A31.svg)](https://aws.amazon.com/s3/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-LLM-8E75B2.svg)](https://ai.google.dev/)

An end-to-end medical diagnostic ecosystem pairing a Q-Learning Reinforcement Learning Preprocessor, a Hierarchical PyTorch Neural Classifier Ensemble, and a Retrieval-Augmented Generation (RAG) Clinical Justification Generator powered by Google Gemini LLM and MongoDB Atlas.

---

## Key Features

* **Reinforcement Learning Preprocessing Agent**:
  * Employs Q-Learning to dynamically optimize contrast (CLAHE), noise reduction (Gaussian/Median Blur), and edge sharpening (filter2D) on medical MRI scans.
  * Evaluates raw scan confidence and applies adaptive transformations to degraded images, recovering up to +13.5% accuracy loss on noisy inputs.
  * Equipped with an adaptive guardrail (>= 90% confidence) to bypass clean scans without introducing image distortion.

* **Hierarchical Classifier Ensemble (98.0% Precision)**:
  * **Generalist Backbone**: ResNet-50 neural network classifying 12 broad organ system categories with 98.0% baseline precision.
  * **7 Specialized Classifiers**: Dedicated domain networks (LiteBrainNet2, InfectiousBrainNet, StableMetabolicNet, NeoplasticBrainNet, MicroLiverNet, CustomLiverNet) pinpointing specific diseases (e.g., Osmotic Demyelination Syndrome, Fukuyama Muscular Dystrophy, Hepatocellular Carcinoma).

* **Agentic RAG Clinical Reporting System**:
  * Combines SentenceTransformers (all-MiniLM-L6-v2) dense vector search with Google Gemini LLM (gemini-1.5-flash) to generate structured clinical justification reports.
  * Cross-references structured medical textbook knowledge with historical patient execution logs.

* **Ultra-Low Latency UI & Asynchronous Engine**:
  * **Two-Stage Generator UI (yield)**: Displays classification scores and preprocessed images in < 0.8 seconds.
  * **Async Multi-Threading (ThreadPoolExecutor)**: Offloads DICOM scan S3 cloud uploads and MongoDB session logging to non-blocking background workers.
  * Full LLM RAG report synthesis completed in < 1.8 seconds.

* **Professional Web Dashboard**:
  * Custom Gradio UI styled with a warm cream background (#F6F3ED), pure white card containers (#FFFFFF), dark typography, and a compact MongoDB status badge.

---

## System Architecture

```mermaid
graph TD
    A[Raw DICOM / Image Upload] --> B[Load Medical Image & Rescale]
    B --> C[Check Initial Generalist Confidence]
    C -->|Confidence >= 90%| D[Identity Passthrough]
    C -->|Confidence < 90%| E[Q-Learning Preprocessing Agent]
    E -->|Apply CLAHE / Blur / Sharpen| F[Enhanced Image Matrix]
    D --> F
    F --> G[Hierarchical PyTorch Hub]
    G --> H[ResNet-50 Generalist Classifier]
    H -->|Confidence >= 90%| I[Route to Disease Specialist Model]
    H -->|Confidence < 90%| J[Abort: Flag for Manual Review]
    I --> K[Specialist Diagnosis & Confidence]
    K --> L[Parallel Vector Search: LightVectorDB]
    L --> M[Google Gemini LLM RAG Synthesis]
    M --> N[Structured Clinical Report Output]
    K --> O[Background Thread: AWS S3 Upload]
    K --> P[Background Thread: MongoDB Audit Logging]
```

---

## System Benchmarks

| Metric | Standard Scans | Degraded / Noisy Scans (No Preprocessing) | With RL Preprocessing Agent |
| :--- | :--- | :--- | :--- |
| **ResNet-50 Generalist Precision** | **98.00%** | 85.20% | **98.00% (Restored)** |
| **System Classification Accuracy** | **98.02%** | 84.52% | **98.02% (Restored)** |
| **Degraded Scans Corrected** | — | — | **+435 scans recovered** |
| **Classification Inference Latency** | ~0.08s | ~0.20s | **Sub-second (< 0.2s)** |
| **Total RAG Synthesis Latency** | ~15.0s (Sequential) | — | **< 1.8 seconds** |

---

## Technology Stack

| Component | Technologies & Libraries |
| :--- | :--- |
| **Frontend UI** | Gradio 4.x/5.x, Custom Vanilla CSS, Base64 Dynamic Logo Loader, SVG |
| **Backend Core** | Python 3.10+, concurrent.futures, PyDICOM, OpenCV (cv2), PIL |
| **Machine Learning** | PyTorch, torchvision (ResNet-50), Q-Learning DQN, SentenceTransformers |
| **Generative AI** | Google Gemini LLM API (google-generativeai) |
| **Databases** | MongoDB Atlas (pymongo), In-Memory Vector DB (LightVectorDB) |
| **Cloud Storage** | AWS S3 (boto3) with 7-day presigned URLs |

---

## Repository Structure

```
SetCONNECT_Git_MedAI/
│
├── app.py                      # Main Gradio application, UI, and async orchestration
├── medical_rag.py              # MedicalRAGPipeline, LightVectorDB, and Gemini setup
├── rl_agent.json               # Trained Q-Table states (1,768 discretized entries)
├── rlagent.wt                  # Trained ResNet-50 Generalist model weights
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment credentials
└── README.md                   # Project documentation
```

---

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/setconnectglobal/Medical-AI.git
cd Medical-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root directory:

```env
# Google Gemini LLM API Key
GEMINI_API_KEY=your_gemini_api_key_here

# MongoDB Atlas Connection URI
MONGO_URI=mongodb+srv://username:password@cluster0.mongodb.net/?retryWrites=true&w=majority

# AWS S3 Cloud Storage Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET=your_s3_bucket_name
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:7860`.

---

## License
Distributed under the MIT License. See `LICENSE` for details.
