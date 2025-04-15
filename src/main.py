import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import pipeline, AutoModel, DistilBertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

pipe = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def map_sentiments(df):
    sentiment_mapping = {
        "Very Negative": 0,
        "Negative": 1,
        "Neutral": 2,
        "Positive": 3,
        "Very Positive": 4
    }
    df['label'] = df['Klasa'].map(sentiment_mapping)
    return df

class Tabu_Architecture(nn.Module):
    def __init__(self, tabu):
        super(Tabu_Architecture, self).__init__()
        self.tabu = tabu
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 5)
    def forward(self, sent_id, mask):
        outputs = self.tabu(sent_id, attention_mask=mask)
        cls_hs = outputs[0][:, 0, :]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = [b.to(device) for b in batch]
        sent_id, mask, labels = batch
        optimizer.zero_grad()
        outputs = model(sent_id, mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = [b.to(device) for b in batch]
            sent_id, mask, labels = batch
            outputs = model(sent_id, mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, output_dict=False, zero_division=0)  
    return total_loss / len(dataloader), report

def fine_tune_model(model, method):
    if method == "train_all":
        # Train the model from scratch (all neurons active)
        for param in model.parameters():
            param.requires_grad = True
    elif method == "freeze_some":
        # Freeze some neurons 
        for name, param in model.named_parameters():
            if "fc" in name:  # Example: only train fully connected layers
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif method == "freeze_all":
        # Freeze all neurons (no training)
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise ValueError("Invalid fine-tuning method. Choose from 'train_all', 'freeze_some', or 'freeze_all'.")

def encode(texts, tokenizer, max_seq_len):
    return tokenizer.batch_encode_plus(
        list(texts),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file_path = "./notebooks/data/Final opinie.xlsx"
    data = load_data(data_file_path)
    data = map_sentiments(data)

    # Set random seed for reproducibility
    SEED = 2025
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Splitting data
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        data['Opinia'], data['label'],
        random_state=SEED,
        test_size=0.3,
        stratify=data['label']
    )
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels,
        random_state=SEED,
        test_size=0.5,
        stratify=temp_labels
    )

    print("Liczba próbek w zbiorach:")
    print(f"Train: {len(train_text)}, Validation: {len(val_text)}, Test: {len(test_text)}")

    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
    max_seq_len = 50

    tokens_train = encode(train_text.tolist(), tokenizer, max_seq_len)
    tokens_val = encode(val_text.tolist(), tokenizer, max_seq_len)
    tokens_test = encode(test_text.tolist(), tokenizer, max_seq_len)

    # Convert Integer Sequences to Tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    # DataLoader setup
    batch_size = 32

    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Model setup
    tabu = AutoModel.from_pretrained("tabularisai/multilingual-sentiment-analysis")
    model = Tabu_Architecture(tabu).to(device)

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    fine_tune_model(model, method="freeze_some")

    # Skip training if all parameters are frozen
    if not any(param.requires_grad for param in model.parameters()):
        print("All parameters are frozen. Skipping training.")
    else:
        best_val_loss = float('inf')
        patience = 7  # Number of epochs to wait
        trigger_times = 0

        # Training loop
        epochs = 100
        for epoch in range(epochs):
            train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
            val_loss, val_report = evaluate(model, val_dataloader, loss_fn, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(val_report)

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), 'best_model.pt')
                best_val_loss = val_loss
                trigger_times = 0
            else:
                scheduler.step(val_loss)
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

    # Test evaluation
    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_report = evaluate(model, test_dataloader, loss_fn, device)
    print("Test Results:")
    print(test_report)