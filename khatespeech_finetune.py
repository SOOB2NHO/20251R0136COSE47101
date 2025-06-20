import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import evaluate

# 1. 파일 로드
df = pd.read_csv("train.csv")

# 2. 다중 클래스 레이블: hate, offensive, none 그대로 사용
label2id = {'hate': 0, 'offensive': 1, 'none': 2}
df_use = df[['comments', 'hate']].rename(columns={'comments': 'text', 'hate': 'label_text'})
df_use['label'] = df_use['label_text'].map(label2id)

# 3. 학습/검증 분할
train_texts, val_texts = train_test_split(df_use, test_size=0.1, stratify=df_use['label'], random_state=42)

# 4. 토크나이저 로드 및 토큰화
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = Dataset.from_pandas(train_texts)
val_dataset = Dataset.from_pandas(val_texts)

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. 모델 로드 (num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 6. 평가 지표
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_strategy="epoch",
    report_to="none"
)

# 8. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9. 학습 시작
trainer.train()

# requirements
# !pip install transformers==4.41.2
# !pip install peft==0.7.1
# !pip install numpy==1.25.2

# 사용법
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("./korean-sentiment-model-2")
# tokenizer = AutoTokenizer.from_pretrained("./korean-sentiment-model-2")