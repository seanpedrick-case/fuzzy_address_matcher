import math
import torch
import torch.nn as nn
import torch.nn.init as init


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        pad_idx,
    ):
        super(TextClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # GRU layers
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirection

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_embedded)

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Pass the entire output tensor to the FC layer for token-level classification
        return self.fc(output)


class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        pad_idx,
    ):
        super(LSTMTextClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layers
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirection

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )

        # Note: LSTM returns both the output and a tuple of (hidden state, cell state)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Pass the entire output tensor to the FC layer for token-level classification
        return self.fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        # If pe doesn't exist or its sequence length is different from x's sequence length
        if not hasattr(self, "pe") or self.pe.size(0) != x.size(1):
            max_len = x.size(1)
            pe = torch.zeros(max_len, self.d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float()
                * (-math.log(10000.0) / self.d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe.to(x.device))

        return x + self.pe[:, : x.size(1), :]


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        init.kaiming_uniform_(m.weight, nonlinearity="relu")


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        num_encoder_layers,
        num_classes,
        dropout,
        pad_idx,
    ):
        super(TransformerClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Transformer with dropout
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            transformer_encoder, num_layers=num_encoder_layers
        )

        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)
        )

    def create_attention_mask(self, src, pad_idx):
        return src == pad_idx

    def forward(self, src, pad_idx):

        # Check pad_idx
        if isinstance(pad_idx, torch.Tensor) and torch.numel(pad_idx) > 1:
            raise ValueError(
                "Expected pad_idx to be a scalar value, but got a tensor with multiple elements."
            )

        # Transpose src to have shape (seq_len, batch_size)
        src = src.transpose(0, 1)

        # Embedding
        x = self.embedding(src)

        # Positional Encoding
        x = self.pos_encoder(x.to(self.device))

        # Create attention mask
        src_key_padding_mask = self.create_attention_mask(
            src.transpose(0, 1), pad_idx
        )  # Transpose back to (batch_size, sequence_length)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # print(model.state_dict())
        # Classification
        return self.classifier(x)
