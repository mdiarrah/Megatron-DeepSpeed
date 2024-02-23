from transformers import pipeline

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Inference with base model
classifier = pipeline("sentiment-analysis",model="bert-base-uncased")
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
output = classifier(text)
print("output: {}".format(output))