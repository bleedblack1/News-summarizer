# config.py

MODEL_NAME = "facebook/bart-base"

MAX_ARTICLE_LEN = 768        # safe for 4GB
MAX_SUMMARY_LEN = 128

TRAIN_SAMPLES = 500          # demo fine-tuning
VAL_SAMPLES = 100
TEST_SAMPLES = 100

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2         # effective batch size = 2
LEARNING_RATE = 3e-5
EPOCHS = 2

SEED = 42
OUTPUT_DIR = "./bart-news-finetuned"
