import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
            
        self.encoded_texts = [self.encode(t) for t in texts]
        
    def build_vocab(self, texts):
        words = [word for text in texts for word in str(text).split()]
        counter = collections.Counter(words)
        vocab = {w: i+2 for i, (w, _) in enumerate(counter.most_common(20000))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab
        
    def encode(self, text):
        words = str(text).split()
        encoded = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]
        if len(encoded) < self.max_len:
            encoded += [self.vocab['<PAD>']] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return encoded
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # output shape: (batch_size, seq_len, hidden_dim*2)
        lstm_out, _ = self.lstm(embedded)
        # Getting the last output representations
        final_state = lstm_out[:, -1, :] 
        out = self.fc(final_state)
        return out

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, filter_sizes=[3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x).unsqueeze(1) 
        # embedded shape: (batch_size, 1, seq_len, embed_dim)
        
        import torch.nn.functional as F
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved[n] shape: (batch_size, num_filters, seq_len - fs + 1)
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[n] shape: (batch_size, num_filters)
        
        cat = torch.cat(pooled, dim=1)
        out = self.fc(cat)
        return out
