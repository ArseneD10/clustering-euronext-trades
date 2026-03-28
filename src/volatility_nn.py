import mlx.nn as nn 
import mlx.core as mx

from .config import EncoderConfig, VolatilityConfig
from .base_encoder_nn import Encoder
from typing import Dict

class VolatilityPredictor(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, volatility_config: VolatilityConfig):
        super().__init__()
        self.encoder = Encoder(encoder_config)
        
        self.pred_head = nn.Sequential(
            nn.Linear(encoder_config.encoder_input_size, volatility_config.hidden_size),
            volatility_config.activation(),
            nn.Dropout(volatility_config.dropout_rate),
            nn.Linear(volatility_config.hidden_size, 1),
            nn.ReLU()
        )

    def __call__(self, categorical_features: Dict[str, mx.ArrayLike], features: mx.ArrayLike):
        encoder_output = self.encoder(features=features, categorical_features=categorical_features).mean(axis=1)
        pred = self.pred_head(encoder_output)
        return pred
