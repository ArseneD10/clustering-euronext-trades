import mlx.nn as nn
import mlx.core as mx

from typing import List, Dict
from .config import EncoderConfig

def unflatten(col, array, unsqueeze=False):
    assert(len(col) == array.shape[-1]), f"Lenght columns--> {len(col)} || Array shape --> {array.shape}"
    unflat = {}
    for idx, c in enumerate(col):
        unflat[c] = mx.array(array[..., idx], dtype=mx.int32)
        if unsqueeze:
            unflat[c] = unflat[c][None, :]
    return unflat

class CategoricalEncoder(nn.Module):
    
    def __init__(self, col:  List[str], embedding_dim: List[int], num_embedding: List[int]):
        super().__init__()
        assert(len(col) == len(embedding_dim) and len(col) == len(num_embedding))
        self.embedding = {k: nn.Embedding(num_embeddings=n, dims=d) for k, n, d in zip(col, num_embedding, embedding_dim)}
        self.embedding_names = col

    def __call__(self, data: Dict[str, mx.ArrayLike])->Dict[str, mx.ArrayLike]:
        out = {}
        for name in list(data.keys()):
            out[name] = self.embedding[name](data[name])
        return out


class AttnBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        n = 2
        while dim % n != 0:
            n += 1
        self.attn = nn.MultiHeadAttention(dims= dim, num_heads=n)
        self.block = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                        nn.LayerNorm(dim)
                    )
        
    def forward_with_activation(self, x, layer_name, hook = {}):
        attn = self.attn(x, x, x)
        x = x + attn
        block = self.block(x)
        
        hook[layer_name+"_attention"] = attn
        hook[layer_name+"_mlp"] = block
        
        return x+block, hook
    
    def __call__(self, x):
        x = x + self.attn(x, x, x)
        return x + self.block(x)
    

class Encoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        max_sequence_size = 200
        self.categorical_encoder = CategoricalEncoder(col=config.categorical_col, embedding_dim=config.embedding_dim, num_embedding=config.num_embedding)
        self.positional_encoder = mx.random.normal(shape=(max_sequence_size, config.encoder_input_size))
        self.layer = [AttnBlock(config.encoder_input_size) for _ in range(config.depth)]

    def forward_with_activation(self, features: mx.ArrayLike, categorical_features: Dict[str, mx.ArrayLike]):  
        categorical = self.categorical_encoder(categorical_features)
        categorical = mx.concatenate([arr for arr in categorical.values()], axis=-1)  
        input_ = mx.concatenate([categorical, features], axis=-1)
        input_ = input_ + self.positional_encoder[: input_.shape[1]]
        hook = {}
        for idx, layer in enumerate(self.layer):
            input_, hook = layer.forward_with_activation(input_, layer_name=f"encoder_block_{idx}", hook=hook)
        return input_, hook


    def __call__(self, features: mx.ArrayLike, categorical_features: Dict[str, mx.ArrayLike]):

        categorical = self.categorical_encoder(categorical_features)
        categorical = mx.concatenate([arr for arr in categorical.values()], axis=-1)
        
        input_ = mx.concatenate([categorical, features], axis=-1)
        input_ = input_ + self.positional_encoder[: input_.shape[1]]

        for layer in self.layer:
            input_ = layer(input_)

        return input_

