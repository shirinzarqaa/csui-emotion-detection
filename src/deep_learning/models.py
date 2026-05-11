import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BiLSTM(nn.Module):
    """Bidirectional LSTM head for sequence embeddings → multi-label logits."""
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.3, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        final = lstm_out[:, -1, :]
        return self.fc(self.dropout(final))


class TextCNN(nn.Module):
    """TextCNN head for sequence embeddings → multi-label logits."""
    def __init__(self, embed_dim, num_classes, num_filters=100, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs, padding=fs // 2) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, embeddings):
        x = embeddings.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.shape[2]).squeeze(2)
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))


class FastTextDataset(Dataset):
    """Dataset that encodes texts as word-index sequences for FastText embedding lookup."""
    def __init__(self, texts, labels, word_to_id, max_len=128):
        self.encoded_texts = []
        for t in texts:
            ids = [word_to_id.get(w, word_to_id.get('<UNK>', 1)) for w in str(t).split()]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids += [word_to_id.get('<PAD>', 0)] * (max_len - len(ids))
            self.encoded_texts.append(ids)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), self.labels[idx]


class BertDataset(Dataset):
    """Dataset that tokenizes texts on-the-fly with an IndoBERT tokenizer."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), self.labels[idx]