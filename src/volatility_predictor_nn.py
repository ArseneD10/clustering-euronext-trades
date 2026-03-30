import mlx.nn as nn
import mlx.core as mx

from .base_encoder_nn import Encoder
from .config import EncoderConfig, DiscreteConfig

from typing import Dict

class DiscreteHead(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, dir_config: DiscreteConfig, encoder_weight = None):
        super().__init__()
        self.encoder = Encoder(encoder_config)

        if encoder_weight:
            self.encoder.load_weights(encoder_weight)
            print("|INFO| Encoder weight successfully load from ", encoder_weight)
        
        self.head = nn.Sequential(
            nn.Linear(encoder_config.encoder_input_size, dir_config.hidden_size),
            dir_config.activation(),
            nn.Dropout(dir_config.dropout_rate),
            nn.Linear(dir_config.hidden_size, dir_config.num_states),
            nn.Softmax()
        )

    def __call__(self, features: mx.ArrayLike, categorical_features: Dict[str, mx.ArrayLike]):
        x = self.encoder(features=features, categorical_features=categorical_features).mean(axis=1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    dummy = mx.random.normal(shape=(3, 12, 12))
    dummy_int = mx.random.randint(shape=(3, 12, 12))
    