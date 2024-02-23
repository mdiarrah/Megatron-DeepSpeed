from transformers import pipeline


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Inference with finetuned model
classifier = pipeline("sentiment-analysis", model="/data/hive-finetuned-bert") 
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
output = classifier(text)
print("output: {}".format(output))
