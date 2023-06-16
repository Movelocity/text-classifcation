import torch
from torch import nn
import pandas as pd
import data_utils
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader

class RNNClassifier(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim, 
        n_layers, 
        bidirectional, 
        dropout
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]
        
        return self.fc(self.dropout(hidden))
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        # Change the order of dimensions: 
        # (batch_size, sequence_length, input_size) -> (sequence_length, batch_size, input_size)
        embedded = embedded.permute(1, 0, 2)  
        
        _, (hidden, _) = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]
        
        return self.fc(self.dropout(hidden))

def evaluate(model, val_loader):
    count = 0
    total = len(val_loader)

    texts = []
    preds = []
    labels = []

    model.eval()
    for batch in val_loader:
        text = batch['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(text).squeeze(1)

        # Get the output logits and convert them to probabilities using the softmax function
        probs = outputs.softmax(-1)
        top_probs, top_indices = probs.topk(k=5, sorted=True)

        for indice_, label_ in zip(top_indices, batch['labels']):
            preds.append(indice_.cpu().numpy())
            labels.append(label_.cpu().numpy())

        for ids_ in text:
            texts.append(tokenizer.decode(ids_, skip_special_tokens=True))

        count += 1
        print(f"\r{count/total*100:.3f} %, {count}", end="", flush=True)

    df = pd.DataFrame({
        "text": texts,
        "label": [int(l) for l in labels],
        "pred1": [int(p[0]) for p in preds],
        "pred2": [int(p[1]) for p in preds],
        "pred3": [int(p[2]) for p in preds],
        "pred4": [int(p[3]) for p in preds],
        "pred5": [int(p[4]) for p in preds],
    })
    df.to_csv("eval_result.csv")

    top1_accuracy = data_utils.top_n_accuracy(df, 1)
    top3_accuracy = data_utils.top_n_accuracy(df, 3)
    top5_accuracy = data_utils.top_n_accuracy(df, 5)

    print(f'Top1 Accuracy: {top1_accuracy:.2f}')
    print(f'Top3 Accuracy: {top3_accuracy:.2f}')
    print(f'Top5 Accuracy: {top5_accuracy:.2f}')


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--log_interval", type=int, default=100, help="interval of batch")
    parser.add_argument("--layers", type=int, default=2, help="RNN Layers")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(f"args: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_labels = data_utils.get_class_labels()

    tokenizer = BertTokenizer.from_pretrained("./bert")
    train_dataset = data_utils.TextClassificationDataset('data/train.txt', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    vocab_size = len(tokenizer.vocab)
    print(f"vocab_size: {vocab_size}")

    # Instantiate the model
    model = RNNClassifier(
        vocab_size, 
        embedding_dim=100, 
        hidden_dim=256,
        output_dim=len(class_labels),
        n_layers=args.layers, 
        bidirectional=True, 
        dropout=0.5
    ).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    losses = []
    cnt = 0
    for epoch in range(args.epochs):
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()

            text = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            cnt += 1
            if cnt % args.log_interval == 0:
                print(f"batch {cnt}\tloss: {loss.item()}")
    with open("train_log.txt", "w", encoding='utf-8') as f:
        for loss in losses:
            f.write(f"{loss:.4f}\n")

    torch.save(model.state_dict(), f"model-{args.layers}.pth")

    test_dataset = data_utils.TextClassificationDataset("data/test.txt", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    evaluate(model, test_loader)

