from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate

# The rest of your code remains the same

# tsv 파일 불러오기
df = pd.read_csv("train.tsv", sep="\t")

# 감정 레이블 생성: hate/offensive → negative, 그 외 → neutral
def map_sentiment(row):
    if row['hate'] == 'hate' or row['hate'] == 'offensive':
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df.apply(map_sentiment, axis=1)
df_use = df[['comments', 'sentiment']].rename(columns={'comments': 'text'})

# 라벨 인코딩

label2id = {'negative': 0, 'neutral': 1}
df_use['label'] = df_use['sentiment'].map(label2id)

# 학습/검증 분할
train_texts, val_texts = train_test_split(df_use, test_size=0.1, stratify=df_use['label'], random_state=42)

model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = Dataset.from_pandas(train_texts)
val_dataset = Dataset.from_pandas(val_texts)

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 'negative', 'neutral'
)


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",        # 에폭마다 평가
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_strategy="epoch",
    report_to="none"  # wandb 등 연결 안 함
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()