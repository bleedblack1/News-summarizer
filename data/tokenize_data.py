from transformers import BartTokenizerFast
from config import MODEL_NAME, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN
from .load_data import load_cnn_dailymail



class NewsTokenizer:
    def __init__(self):
        self.tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_batch(self, batch):
        # Encoder inputs
        inputs = self.tokenizer(
            batch["article"],
            max_length=MAX_ARTICLE_LEN,
            truncation=True,
            padding="max_length"
        )

        # Decoder targets (labels)
        labels = self.tokenizer(
            batch["highlights"],
            max_length=MAX_SUMMARY_LEN,
            truncation=True,
            padding="max_length"
        )

        inputs["labels"] = labels["input_ids"]
        return inputs



if __name__ == "__main__":
    train_ds, _, _ = load_cnn_dailymail()

    tokenizer = NewsTokenizer()
    sample = tokenizer.tokenize_batch(train_ds[:2])

    print("Keys:", sample.keys())
    print("Input IDs shape:", len(sample["input_ids"][0]))
    print("Labels shape:", len(sample["labels"][0]))
