from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
import wandb


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# define wandb parameters
wandb.login(key="0bd3b89d4bbaa98b6011cc062e7a757da2e3645c")
wandb.init(project="bert-training-rtx4090-zero1-1gpu")

# Define training args and enable deepspeed 
training_args = TrainingArguments("test_trainer",report_to="wandb",evaluation_strategy="epoch", deepspeed="/home/deepspeed/Megatron-DeepSpeed/hive/ds_config_zero1.json")

# Define class labels
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Load the model for text classification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, id2label=id2label, label2id=label2id)

# Load labeled movie review dataset
raw_datasets = load_dataset("imdb")

# Show a labeled training sample
print("sample[0]: ".format(raw_datasets["train"][0]))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Train only on 5000 samples
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5000))

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics, 
)
result = trainer.train()
print(result)

# Save the finetuned Model and its tokenizer
trainer.save_model("/data/hive-finetuned-bert")
tokenizer.save_pretrained("/data/hive-finetuned-bert")