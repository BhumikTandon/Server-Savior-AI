# Server-Savior-AI 🛡️
> **An Unsupervised AI Intrusion Detection System (IDS) powered by Small Language Models.**

**Created by:** Bhumik Tandon

## 📖 Overview
Server-Savior-AI is a lightweight security tool that detects anomalies in server logs (like Brute Force attacks) without needing labeled training data. Instead of using predefined rules (regex), it uses a **Small Language Model (DistilBERT)** trained from scratch to learn the "language" of normal server operations.

When the model encounters an attack pattern it hasn't seen before, its "perplexity" score spikes, triggering a real-time alert.

## 🚀 Key Features
* **Unsupervised Learning:** No need to label "attacks." It learns normality and flags deviations.
* **Privacy-First:** Uses `Drain3` log parsing to mask sensitive IPs and timestamps before processing.
* **Lightweight:** Runs efficiently on CPU using a compact, custom-trained Transformer model.
* **Real-Time:** `guard_dog.py` monitors logs as they are written, acting as a live watchdog.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Hugging Face Transformers** (DistilBERT)
* **PyTorch**
* **Drain3** (Log Parsing)

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
