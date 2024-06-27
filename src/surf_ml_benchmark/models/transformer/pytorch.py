import click

import numpy as np
from typing import Any, Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.modules import CrossEntropyLoss
import torch.optim as optim

# import torch.utils.data as data
import math

from torch.optim.adam import Adam
# import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: str = "cpu") -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k: int = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )  # Query transformation
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, device=device)  # Key transformation
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, device=device)  # Value transformation
        self.W_o = nn.Linear(
            in_features=d_model, out_features=d_model, device=device
        )  # Output transformation

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        # Calculate attention scores
        attn_scores: Tensor = torch.matmul(input=Q, other=K.transpose(-2, -1)) / math.sqrt(self.d_k)  # pyright: ignore[reportRedeclaration]

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores: Tensor = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs: Tensor = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output: Tensor = torch.matmul(input=attn_probs, other=V)
        return output

    def split_heads(self, x: Tensor) -> Tensor:
        # Reshape the input to have num_heads for multi-head attention
        # batch_size, seq_length, d_model = x.size()
        batch_size, seq_length, _ = x.size()

        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: Tensor):
        # Combine the multiple heads back to original shape
        # batch_size, _, seq_length, d_k = x.size()
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:  # pyright: ignore[reportRedeclaration]
        # Apply linear transformations and split heads
        Q: Tensor = self.split_heads(x=self.W_q(Q))  # pyright: ignore[reportConstantRedefinition]
        K: Tensor = self.split_heads(x=self.W_k(K))  # pyright: ignore[reportConstantRedefinition]
        V: Tensor = self.split_heads(x=self.W_v(V))  # pyright: ignore[reportConstantRedefinition]

        # Perform scaled dot-product attention
        attn_output: Tensor = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(x=attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: str = "cpu") -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff, device=device)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model, device=device)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, device: str = "cpu") -> None:  # pyright: ignore[reportRedeclaration]
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        pe: Tensor = torch.zeros(max_seq_length, d_model, device=device)
        position: Tensor = torch.arange(0, max_seq_length, dtype=torch.float, device=device).unsqueeze(dim=1)
        div_term: Tensor = torch.exp(
            input=torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(input=position * div_term)
        pe[:, 1::2] = torch.cos(input=position * div_term)

        self.register_buffer(name="pe", tensor=pe.unsqueeze(dim=0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, device: str = "cpu") -> None:
        super(EncoderLayer, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # pyright: ignore[reportRedeclaration]
        attn_output = self.self_attn(x, x, x, mask)
        x: Tensor = self.norm1(x + self.dropout(attn_output))  # pyright: ignore[reportRedeclaration]
        ff_output = self.feed_forward(x)
        x: Tensor = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, device: str = "cpu") -> None:
        super(DecoderLayer, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.self_attn = MultiHeadAttention(d_model, num_heads, device=device)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, device=device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device=device)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.norm3 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, enc_output: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:  # pyright: ignore[reportRedeclaration]
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x: Tensor = self.norm1(x + self.dropout(attn_output))  # pyright: ignore[reportRedeclaration]
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x: Tensor = self.norm2(x + self.dropout(attn_output))  # pyright: ignore[reportRedeclaration]
        ff_output = self.feed_forward(x)
        x: Tensor = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
        device: str = "cpu",
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=d_model, device=device
        )
        self.decoder_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=d_model, device=device
        )
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            modules=[DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(in_features=d_model, out_features=tgt_vocab_size, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def generate_mask(self, src: Tensor, tgt: Tensor, device: str = "cpu") -> Tuple[Any, Any]:
        src_mask: Tensor = (src != 0).unsqueeze(dim=1).unsqueeze(dim=2)
        tgt_mask: Tensor = (tgt != 0).unsqueeze(dim=1).unsqueeze(dim=3)  # pyright: ignore[reportRedeclaration]
        seq_length: int = tgt.size(1)
        nopeak_mask: Tensor = (
            1 - torch.triu(input=torch.ones(1, seq_length, seq_length, device=device), diagonal=1)
        ).bool()
        tgt_mask: Tensor = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


@click.command()
@click.option("--gpu", is_flag=True, help="Run on gpu")
@click.option("--d_model", help="d_model")
@click.option("--num_heads", help="number of heads")
@click.option("--num_layers", help="Number of layers")
@click.option("--d_ff", help="d_ff")
@click.option("--max_sequence_length", help="Max sequence length.")
@click.option("--dropout", help="Dropout")
@click.option("--src_vocab_size", help="Source vacubulary size.")
@click.option("--tgt_vocab_size", help="Target vacubulary size.")
def cli(**kwargs: str | bool) -> None:
    """Transformer using the Pytorch framework. Runs on the cpu by default."""
    device: str = "cpu"

    if kwargs["gpu"]:
        if not torch.cuda.is_available():
            click.echo(message="GPU was selected, but CUDA environment is missing. Switching to CPU instead.")
        else:
            device = "cuda"
            click.echo(message=torch.cuda.get_device_name(device=0))

    src_vocab_size: int = 5000
    tgt_vocab_size: int = 5000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 100
    dropout: float = 0.1

    torch.manual_seed(seed=42)  # pyright: ignore[reportUnknownMemberType]
    np.random.seed(seed=42)

    transformer: Transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    # Generate random sample data
    src_data: Tensor = torch.randint(
        1, src_vocab_size, (64, max_seq_length), device=device
    )  # (batch_size, seq_length)
    tgt_data: Tensor = torch.randint(
        1, tgt_vocab_size, (64, max_seq_length), device=device
    )  # (batch_size, seq_length)

    criterion: CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=0)
    optimizer: Adam = optim.Adam(params=transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
