import torch
from transformers import (
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from data.load_data import load_cnn_dailymail
from data.tokenize_data import NewsTokenizer
from config import *


def main():
    # Load data
    train_ds, val_ds, _ = load_cnn_dailymail()

    tokenizer_wrapper = NewsTokenizer()
    tokenizer = tokenizer_wrapper.tokenizer

    train_tok = train_ds.map(
        tokenizer_wrapper.tokenize_batch,
        batched=True,
        remove_columns=train_ds.column_names
    )

    # Load model
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    #  MEMORY SAVERS
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,

        fp16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()


