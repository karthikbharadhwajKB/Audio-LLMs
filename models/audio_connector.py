import torch 
import torch.nn as nn
from typing import Optional, List, Dict, Any
import logging 


# Set up logging
logger = logging.getLogger(__name__)

class AudioConnector(nn.Module):
    """
    Trainable connector module between audio encoder and language model.

    Features:
        - Temporal Compression via Conv1D layers
        - Feature dimension Projection
        - Layer Normalization for stable training
    """
    def __init__(
            self, 
            whisper_dim: int = 768,
            llm_dim: int = 4096,
            compression_factor: int = 4,
            dropout: float = 0.1,
    ):
        """
        Intializes the AudioConnector module.

        Args: 
            whisper_dim (int): Dimension of the audio encoder output.
            llm_dim (int): Dimension of the language model input.
            compression_factor (int): Factor by which to compress the temporal dimension.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.whispher_dim = whisper_dim
        self.llm_dim = llm_dim
        self.compression_factor = compression_factor
        self.dropout = dropout

        # Temporal Compression Layer
        self.temporal_conv = nn.Conv1d(
            in_channels=whisper_dim,
            out_channels=whisper_dim,
            kernel_size=compression_factor,
            stride=compression_factor,
            padding=0,
        )

        # Feature Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(whisper_dim, llm_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(llm_dim // 2, llm_dim),
            nn.LayerNorm(llm_dim),
        )


        # Initialize weights
        self._init_weights()

        logger.info(
            f"AudioConnector: {whisper_dim} -> {llm_dim},"
            f" Compression Factor: {compression_factor}x"
        )

    def _init_weights(self):
        """ Intializes connector weights with Xavier/Kaiming Initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear): 
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AudioConnector.

        Args:
            input_features (torch.Tensor): Input features from the audio encoder.
                Shape: (batch_size, seq_len, whisper_dim)
        
                
        Returns:
            torch.Tensor: Transformed features for the language model.
                Shape: (batch_size, compressed_seq_len, llm_dim)
        """
        batch_size, seq_len, hidden_dim = input_features.shape

        # Transpose for Conv1D: (batch_size, hidden_dim, seq_len)
        x = input_features.transpose(1, 2)

        # Apply Temporal Compression
        x = self.temporal_conv(x) # Shape: (batch_size, whisper_dim, compressed_seq_len)

        # Transpose back: (batch_size, compressed_seq_len, whisper_dim)
        x = x.transpose(1, 2)

        # Apply Feature Projection
        llm_embeddings = self.projection(x) # Shape: (batch_size, compressed_seq_len, llm_dim)

        return llm_embeddings