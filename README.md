#  News Summarizer using Fine-Tuned BART

This project implements an **end-to-end news summarization system**
using a **Transformer-based model (BART)**.

It covers the complete machine learning pipeline:
**dataset loading → tokenization → fine-tuning → inference → deployment**.

---

##  Live Demo (Public)

 **Hugging Face Space**  
https://huggingface.co/spaces/NISHANT-INDIA/news-summarizer

Users can:
- Paste a full news article and get a concise summary
- Paste a news article URL (auto-scraped)
- Use the app without login (public access)

---

##  Model Information

- **Base Model**: facebook/bart-base
- **Task**: Abstractive News Summarization
- **Dataset**: CNN / DailyMail
- **Framework**: PyTorch + Hugging Face Transformers

 **Fine-Tuned Model on Hugging Face Hub**  
https://huggingface.co/NISHANT-INDIA/bart-news-finetuned

> Model weights are hosted on Hugging Face Hub to keep the GitHub repository clean.

---

##  Project Structure

##  Project Structure

```text
news-summarizer/
│
├── app/
│   ├── app.py            # Gradio web application
│   ├── scraper.py        # URL → article text extraction
│   └── __init__.py
│
├── data/
│   ├── load_data.py      # CNN/DailyMail dataset loading
│   ├── tokenize_data.py  # Tokenization logic
│   └── __init__.py
│
├── train.py              # Fine-tuning script
├── infer.py              # Inference script
├── main.py               # Experiment entry point
├── config.py             # Training configuration
├── requirements.txt
├── README.md
└── .gitignore

