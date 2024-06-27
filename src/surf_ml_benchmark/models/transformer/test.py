import click
import torch
import torch.nn as nn
from torch.nn.modules import CrossEntropyLoss
import torch.optim as optim
from torch.functional import Tensor

import numpy as np
from torch.optim.adam import Adam


# Define the model
class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.model_type = "Transformer"

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(in_features=model_dim, out_features=output_dim)

    def forward(
        self,
        src: Tensor,  # pyright: ignore[reportRedeclaration]
        tgt: Tensor,  # pyright: ignore[reportRedeclaration]
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        src: Tensor = self.embedding(src)
        tgt: Tensor = self.embedding(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.fc_out(output)

        return output


@click.command()
def cli() -> None:
    torch.manual_seed(seed=42)  # pyright: ignore[reportUnknownMemberType]
    np.random.seed(seed=42)
    # Parameters
    input_dim: int = 5000  # Size of the input vocabulary
    output_dim: int = 5000  # Size of the output vocabulary

    model_dim: int = 512  # Embedding dimensions
    nhead: int = 8  # Number of attention heads
    num_encoder_layers: int = 6  # Number of encoder layers
    num_decoder_layers: int = 6  # Number of decoder layers
    dim_feedforward: int = 2048  # Dimensions of the feedforward network
    dropout: float = 0.1  # Dropout rate

    # Create model
    model: TransformerModel = TransformerModel(
        input_dim,
        model_dim,
        output_dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    )

    # Loss and optimizer
    criterion: CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: Adam = optim.Adam(params=model.parameters(), lr=0.001)

    # Sample data (batch_size, sequence_length)
    src: Tensor = torch.randint(0, input_dim, (64, 100))  # Source sequence
    tgt: Tensor = torch.randint(0, output_dim, (64, 100))  # Target sequence

    # Forward pass
    output = model(src, tgt)

    # Compute loss
    tgt_out: Tensor = tgt.view(-1)  # Flatten target tensor
    output = output.view(-1, output_dim)  # Flatten output tensor for loss computation
    loss = criterion(output, tgt_outt)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt)

        # Compute loss
        tgt_out: Tensor = tgt.view(-1)  # Flatten target tensor
        output = output.view(-1, output_dim)  # Flatten output tensor for loss computation
        loss = criterion(output, tgt_out)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
