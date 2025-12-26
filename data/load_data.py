from datasets import load_dataset

from config import TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, SEED


def load_cnn_dailymail():
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_ds = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
    val_ds   = dataset["validation"].shuffle(seed=SEED).select(range(VAL_SAMPLES))
    test_ds  = dataset["test"].shuffle(seed=SEED).select(range(TEST_SAMPLES))

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train, val, test = load_cnn_dailymail()
    print(train)
    print(val)
    print(test)

