import torch
import gradio as gr
from transformers import BartTokenizerFast, BartForConditionalGeneration
from app.scraper import scrape_article

MODEL_DIR = "bart-news-finetuned"

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def summarize_text(article):
    if not article or len(article.strip()) < 50:
        return "Please enter a longer article."

    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_url(url):
    if not url.startswith("http"):
        return "Please enter a valid URL."

    article = scrape_article(url)
    if not article:
        return "Could not extract article text from this URL."

    return summarize_text(article)


with gr.Blocks(title="News Summarizer") as demo:
    gr.Markdown("##  News Summarizer (Fine-Tuned BART)")

    with gr.Tab("Paste Article Text"):
        text_input = gr.Textbox(lines=15, label="Article Text")
        text_output = gr.Textbox(lines=6, label="Summary")
        gr.Button("Summarize Text").click(
            summarize_text, text_input, text_output
        )

    with gr.Tab("Paste Article URL"):
        url_input = gr.Textbox(label="Article URL")
        url_output = gr.Textbox(lines=6, label="Summary")
        gr.Button("Summarize URL").click(
            summarize_url, url_input, url_output
        )

demo.launch()
