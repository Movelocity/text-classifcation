import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import TextClassificationDataset

if __name__ == "__main__":
    with open('./data/class.txt', 'r', encoding='utf-8') as f:
        class_labels = f.read().strip().split('\n')

    tokenizer = BertTokenizer.from_pretrained("./bert")
    model = BertForSequenceClassification.from_pretrained("./bert", num_labels=len(class_labels))
    model.eval()
    print("Model loaded.")

    test_dataset = TextClassificationDataset('./data/test.txt', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("Data loaded.")
    # Iterate over the DataLoader to get the input batch
    # for batch in test_loader:
        # batch = {k: v.to(model.device) for k, v in batch.items()}

    count = 6
    for sample in test_loader:
        with torch.no_grad():
            outputs = model(**sample)

        # Get the output logits and convert them to probabilities using the softmax function
        probs = outputs.logits.softmax(-1)
        top_probs, top_indices = probs.topk(k=5, sorted=True)
        top_classes = {}
        for i in range(5):
            class_index = top_indices[0][i].item()
            class_prob = top_probs[0][i].item()
            class_name = class_labels[class_index]
            top_classes[class_name] = class_prob

        true_label = class_labels[sample["labels"][0].item()]
        # Get the text and true label from the sample
        text = tokenizer.decode(sample["input_ids"][0], skip_special_tokens=True)

        print(f"Text: {text}; [{true_label}]")
        print(f"Prediction: ")
        for k, v in top_classes.items():
            print(f"  {k}: {v:.4f}")
        print("")

        count -= 1
        if count <= 0:
            break

