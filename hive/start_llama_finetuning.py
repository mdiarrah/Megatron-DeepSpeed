from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
import wandb
from torch.utils.data import DataLoader
from transformers import default_data_collator



metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(sample):
    instruction = tokenizer(sample["instruction"])["input_ids"]
    return tokenizer(sample["output"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]


# define wandb parameters
wandb.login(key="0bd3b89d4bbaa98b6011cc062e7a757da2e3645c")
wandb.init(project="llama7b-finetuning")

# Load labeled movie review dataset
raw_datasets = load_dataset("tatsu-lab/alpaca")
train_ds_packed = raw_datasets["train"]
#eval_ds_packed  = raw_datasets["eval"]

# Show a labeled training sample
print("sample[0]: ".format(raw_datasets["train"][0]))
'''
batch_size = 8  # I have an A100 GPU with 40GB of RAM 😎
train_dataloader = DataLoader(
    train_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator, # we don't need any special collator 😎
)
'''

# Load the model for text classification
model = AutoModel.from_pretrained("huggyllama/llama-7b")

# Define training args and enable deepspeed 
training_args = TrainingArguments("test_trainer",report_to="wandb", deepspeed="/home/deepspeed/Megatron-DeepSpeed/examples_deepspeed/finetune_hf_llama/ds_config_default.json")

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Train only on 1000 samples
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=small_train_dataset, 
)
result = trainer.train()
print(result)

# Save the finetuned Model and its tokenizer
trainer.save_model("/data/hive-finetuned-llama")
tokenizer.save_pretrained("/data/hive-finetuned-llama")