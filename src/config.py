import mlx.nn as nn
import mlx.optimizers as optim
from typing import List

class EncoderConfig:

    def __init__(self, num_continuous_col: int, 
                 depth: int,
                 embedding_dim: List[int], 
                 num_embedding: List[int], 
                 categorical_col: List[str]):
        
        self.num_cat_col = len(categorical_col)
        self.num_continuous_col = num_continuous_col
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.categorical_col = categorical_col
        self.depth = depth

        self.encoder_input_size = num_continuous_col + sum(embedding_dim)


class VolatilityConfig:

    def __init__(self, hidden_size: int = 32, activation = nn.ReLU, dropout_rate: float = 0.2):
    
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_rate = dropout_rate

class DiscreteConfig:

    def __init__(self, num_states: int = 3, hidden_size: int = 32, activation = nn.ReLU, dropout_rate: float = 0.2):
    
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.num_states = num_states

class VAEConfig:

    def __init__(self, compression_ratio: int = 3, hidden_size: int = 32, activation = nn.ReLU, dropout_rate: float = 0.2):
    
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.compression_ratio = compression_ratio


class OptimizerConfig:

    def __init__(self, optimizer_name: str, lr: float):

        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(learning_rate=lr)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(learning_rate=lr)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(learning_rate=lr)
        else:
            raise ValueError(f"Couldn't recognize {optimizer_name}")