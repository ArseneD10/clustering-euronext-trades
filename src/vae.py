import mlx.nn as nn
import mlx.core as mx

from .base_encoder_nn import Encoder
from .config import EncoderConfig, VAEConfig

from typing import Dict


class VAE(nn.Module):

    def __init__(self, encoder_config: EncoderConfig, vae_config: VAEConfig):
        super().__init__()
        self.encoder_config = encoder_config
        self.feature_encoder = Encoder(encoder_config)
        self.latent_dim = int(encoder_config.encoder_input_size // vae_config.compression_ratio)
    
        self.base = nn.Sequential(
            nn.Linear(encoder_config.encoder_input_size, vae_config.hidden_size),
            vae_config.activation(),
            )
  
        self.mean = nn.Sequential(
            nn.Linear(vae_config.hidden_size, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        self.logvar = nn.Sequential(
            nn.Linear(vae_config.hidden_size, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, vae_config.hidden_size),
            vae_config.activation(),
            nn.Dropout(vae_config.dropout_rate),
            nn.Linear(vae_config.hidden_size, encoder_config.encoder_input_size)
        )
        
    def _unembedding(self, categorial_input: mx.ArrayLike):
        # Split by embedding dim
        # Categorical output shape --> (B, S, N) N -> Sum of embedding dim
        embedding_dim = self.encoder_config.embedding_dim
        embedding_cols = self.encoder_config.categorical_col
        categorial_output = {}
        assert(categorial_input.shape[-1]) == sum(embedding_dim), f"Categorial model output doesn't match with encoder input, got {categorial_input.shape} | {sum(embedding_dim)}"
        
        start_indices = [0] 
        for dim in embedding_dim:
            ids = start_indices[-1] + dim
            start_indices.append(ids)
        start_indices.pop()

        for idx, name, dim in zip(start_indices, embedding_cols, embedding_dim):
            arr = categorial_input[..., idx: idx+ dim]
            probs = mx.softmax(arr, axis=-1)
            categorical = probs.max(axis=-1)
            categorial_output[name] = categorical
        
        return categorial_output

    def _reparametrize(self, hidden_states: mx.ArrayLike):    
        mean = self.mean(hidden_states)
        if self._training:
            logvar = self.logvar(hidden_states)
            eps = mx.random.normal(shape=logvar.shape)
            return mean + logvar * eps
        
        return mean
    
    def encode_latent(self, cfeatures: mx.ArrayLike, categorical_features: Dict[str, mx.ArrayLike]):
        embedding = self.feature_encoder(features=cfeatures, categorical_features=categorical_features)
        hidden_states = self.base(embedding)
        latent = self._reparametrize(hidden_states)
        return latent
    
    def decode_latent(self, latent: mx.ArrayLike):
        decode = self.decoder(latent)
        categorical_lenght = sum(self.encoder_config.embedding_dim)
        categorical_decode = decode[..., :categorical_lenght]
        categorical_decode = self._unembedding(categorical_decode)
        continous_decode = decode[..., categorical_lenght :]
        return continous_decode, categorical_decode

    def __call__(self, features: mx.ArrayLike, categorical_features: Dict[str, mx.ArrayLike]):
        latent = self.encode_latent(features,categorical_features=categorical_features)
        return self.decode_latent(latent)
    
