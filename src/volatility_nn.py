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

    def forward_with_activation(self, categorical_features: Dict[str, mx.ArrayLike], features: mx.ArrayLike):
        embedding, hook = self.encoder.forward_with_activation(features=features, categorical_features=categorical_features)
        for idx, layer in enumerate(self.pred_head.layers):
            embedding = layer(embedding)
            hook[f"pred_head_layer_{idx}"] = embedding
        return embedding, hook

    def __call__(self, categorical_features: Dict[str, mx.ArrayLike], features: mx.ArrayLike):
        encoder_output = self.encoder(features=features, categorical_features=categorical_features).mean(axis=1)
        pred = self.pred_head(encoder_output)
        return pred

class VariationalVolatilityPredictor(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, volatility_config: VolatilityConfig):
        super().__init__()
        self.encoder = Encoder(encoder_config)
        
        self.logvar = nn.Sequential(
            nn.Linear(encoder_config.encoder_input_size, volatility_config.hidden_size),
            volatility_config.activation(),
            nn.Dropout(volatility_config.dropout_rate),
            nn.Linear(volatility_config.hidden_size, 1),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(encoder_config.encoder_input_size, volatility_config.hidden_size),
            volatility_config.activation(),
            nn.Dropout(volatility_config.dropout_rate),
            nn.Linear(volatility_config.hidden_size, 1),
            nn.ReLU()
        )

    def forward_with_activation(self, categorical_features: Dict[str, mx.ArrayLike], features: mx.ArrayLike):
        embedding, hook = self.encoder.forward_with_activation(features=features, categorical_features=categorical_features)
        for idx, layer in enumerate(self.pred_head.layers):
            embedding = layer(embedding)
            hook[f"pred_head_layer_{idx}"] = embedding
        return embedding, hook

    def _reparametrize(self, input_):
        mu = self.mu(input_)
        if self.training:
            std = mx.exp(self.logvar(input_)**0.5)
            mu = mu + std * mx.random.normal(shape=(std.shape))
        return mu
    
    def __call__(self, categorical_features: Dict[str, mx.ArrayLike], features: mx.ArrayLike):
        encoder_output = self.encoder(features=features, categorical_features=categorical_features).mean(axis=1)
        pred = self._reparametrize(encoder_output)
        return pred
