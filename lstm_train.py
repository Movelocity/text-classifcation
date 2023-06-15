import torch
from torch import nn

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


def train(model, iterator, criterion, optimizer):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        optimizer.zero_grad()

        text = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./data/class.txt', 'r', encoding='utf-8') as f:
        class_labels = f.read().strip().split('\n')

    vocab_size = len(tokenizer.vocab)
    print(f"vocab_size: {vocab_size}")

    # Instantiate the model
    model = RNNClassifier(
        vocab_size, 
        embedding_dim=100, 
        hidden_dim=256, 
        output_dim=len(class_labels),
        n_layers=2, 
        bidirectional=True, 
        dropout=0.5
    ).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss = train(model, test_loader, criterion, optimizer)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')

