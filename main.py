# main.py

from data.load_data import load_cnn_dailymail
from data.tokenize_data import NewsTokenizer


def main():
    # 1. Load data
    train_ds, val_ds, test_ds = load_cnn_dailymail()

    # 2. Load tokenizer
    tokenizer = NewsTokenizer()

    # 3. Tokenize datasets
    train_tok = train_ds.map(
        tokenizer.tokenize_batch,
        batched=True,
        remove_columns=train_ds.column_names
    )

    val_tok = val_ds.map(
        tokenizer.tokenize_batch,
        batched=True,
        remove_columns=val_ds.column_names
    )

    # 4. Sanity check
    sample = train_tok[0]

    print("ENCODER INPUT (decoded):")
    print(tokenizer.tokenizer.decode(sample["input_ids"][:200]))

    print("\nTARGET SUMMARY (decoded):")
    print(tokenizer.tokenizer.decode(
        [t for t in sample["labels"] if t != tokenizer.tokenizer.pad_token_id]
    ))


if __name__ == "__main__":
    main()
