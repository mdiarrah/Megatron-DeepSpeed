from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric


class HFTrainer(Trainer):
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        # Hack to fix: https://github.com/huggingface/transformers/issues/24558
        if self.args.auto_find_batch_size:
            self.model_wrapped = self.model
            self.deepspeed = None
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


#training_args = TrainingArguments("test_trainer", deepspeed="/home/deepspeed/Megatron-DeepSpeed/hive/ds_config_zero3.json")
training_args = TrainingArguments("test_trainer",evaluation_strategy="epoch", deepspeed="/home/deepspeed/Megatron-DeepSpeed/hive/ds_config_zero3.json")
#model = AutoModel.from_pretrained("bert-base-cased")#, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#tokenizer = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=small_train_dataset, 
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
#trainer.evaluate()
result = trainer.train()
print(result)
trainer.save_model("/data/bert-ft")