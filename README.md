# 🔬 PathoAgent — Predictive Health Risk Agent

> *Your lab report knows your future. Powered 100% by free, local AI.*

PathAgent analyzes blood test reports by reasoning over **combinations of lab markers** to predict future disease risks — not just flag individual abnormal values.

## 🆓 Completely Free Stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Llama Phi3 via Ollama | FREE |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | FREE |
| Vector DB | ChromaDB (local) | FREE |
| PDF Parsing | PyMuPDF | FREE |
| UI | Streamlit | FREE |

**No API keys. No cloud costs. Runs 100% on your machine.**

## 🧠 What makes it unique

Most lab report tools analyze values **one by one**. PathAgent identifies **risk clusters** — patterns across multiple markers that together signal a developing disease — and generates a forward-looking risk score with cited medical evidence.

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Node 1: Extraction Agent      → Pulls all lab values from PDF
    │
    ▼
Node 2: Anomaly Detection     → Flags out-of-range markers
    │
    ▼
Node 3: Combination Reasoning → Identifies disease risk clusters ← UNIQUE
    │
    ▼
Node 4: Risk Scoring          → HIGH/MEDIUM/LOW + time horizon
    │
    ▼
Node 5: Action Plan           → Patient summary + Doctor summary
```

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Install Ollama & pull Llama 3.1
```bash
# Download from https://ollama.com then:
ollama pull llama3.1
ollama serve
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment
```bash
cp .env.example .env
# No API keys needed — just verify the defaults are correct
```

### 4. Add medical guideline PDFs to data/guidelines/
Free sources:
- https://diabetesjournals.org/care
- https://www.who.int/publications
- https://medlineplus.gov

### 5. Ingest PDFs into ChromaDB (run once)
```bash
python rag/ingest.py
```

### 6. Run the app
```bash
streamlit run app.py
```

## 📁 Folder Structure

```
pathagent/
├── app.py                    ← Streamlit UI
├── agent/
│   ├── embeddings.py         ← sentence-transformers (FREE local embeddings)
│   ├── graph.py              ← LangGraph graph
│   ├── nodes.py              ← All 5 agent nodes (uses Llama 3.1)
│   ├── state.py              ← Shared state
│   └── tools.py              ← ChromaDB retrieval
├── rag/
│   └── ingest.py             ← Load guidelines into ChromaDB
├── utils/
│   └── pdf_parser.py         ← PDF text extraction
├── data/
│   └── guidelines/           ← Place PDFs here
└── requirements.txt
```

## ⚠️ Disclaimer
PathAgent is a research prototype and does NOT provide medical advice. Always consult a qualified physician.
