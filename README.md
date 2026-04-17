# The AI Viral Architect

An end-to-end agentic AI system that analyses YouTube trending video data to generate complete content briefs for social media creators — including trend insights, 5 hook variations, and a full script outline.

---

## System Architecture

![Architecture Diagram](assets/architecture_diagram.png)

---

## Project Structure

| Notebook | Description |
|---|---|
| `01_transcripts.ipynb` | Identifies top 500 popular videos by engagement ratio and extracts transcripts using OpenAI Whisper |
| `02_merge_transcripts.ipynb` | Merges transcripts collected independently by 3 team members (~150 each), fills missing columns |
| `03_hook_extraction.ipynb` | Extracts hook segments (first 10s) and body segments (10–60s) from transcripts |
| `04_rag_pipeline.ipynb` | Chunks transcripts, generates embeddings (all-MiniLM-L6-v2), builds FAISS index (754 vectors) |
| `05_agents_and_output.ipynb` | Full 5-agent pipeline with content generation, rule-based scoring, and LLM evaluation |

---

## How to Run

These notebooks were developed on **Kaggle** with a T4 GPU.
To reproduce:

1. Upload `top500_popular.csv` from the `data/` folder as a Kaggle dataset
2. Upload the FAISS index and chunks metadata (outputs of `04_rag_pipeline.ipynb`) as a Kaggle dataset
3. Add your **Groq API key** to Kaggle Secrets under the name `api`
4. Run notebooks in order: 01 → 02 → 03 → 04 → 05

---

## Tech Stack

| Component | Technology |
|---|---|
| Dataset | YouTube Trending Video Dataset (US + GB) |
| Transcripts | OpenAI Whisper + youtube-transcript-api |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (IndexFlatIP, 754 chunks) |
| LLM | Llama 3.3 70B via Groq API |
| Fine-tuning | LoRA on TinyLlama-1.1B (PEFT, r=8) |
| Evaluation | Rule-based scorer + LLM-as-judge (Llama 3.3 70B) |

---
