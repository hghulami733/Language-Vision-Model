from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings:
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" # This indicates no padding is added
        )

        self.num_patchs = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patchs
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_size, Channels, Height, Width]
        # Convolve the "patch_size" kernel over the image, with no overlapping patches since the stride is equal
        # the output of the convolution will have shape [Batch_size, Embed_dim, Num_patches_h, Num_patches_w]
        # where Num_patchs = height // patch_size and Num_patches_w // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        """ [Batch_size, Embed_dim, Num_patches_h, Num_patches_w] -> [Batch_size, Embed_dime, Num]
        where Num_patches = Num_patches_h * Num_patches_w
        """
        embeddings = patch_embeds.flatten(2)
        # [Batch_size, Embed_dim, Num_patches_h, Num_patches_w] -> [Batch_size, Num_patches, ]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch, Each positional embedding is a vector of size [Embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_size, Num_patches, Embed_dim]
        return embeddings
    
    
class SiglipAttention(nn.Module):
    """ Multi-headed attention from 'Attention is All you need' paper """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states [Batch_size, Num_patches, Embed_dim] 
        batch_size, seq_len = hidden_states.size()

        # query_states: [Batch_size, Num_patches, Embed_dim]
        query_states = self.q_proj(hidden_states)

        # key_states : [Batch_size, Num_patches, Embed_dim] 
        key_states = self.k_proj(hidden_states)

        # value_states : [Batch_size, Num_patches, Embed_dim] 
        value_states = self.v_proj(hidden_states)

        # query_states : [Batch_size, Num_heads, Num_patches, Head_dim] * [Batch_size, Num_heads, Head_dim, Num_patches]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention using the formula Q * K ^ T / sqrt(d_k). attn_weights: [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weight = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weight.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weight should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is "
                f"{(attn_weight.size())}"
            )
        
        # Apply the softmax row-wise. attn_weights: [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout only during training
        attn_weight = nn.functional.dropout(attn_weight, p=self.dropout, training=self.training)

        # Multiply the attention weights by the value states. attn_output: [Batch_size, Num_heads, Num_patches, Head_dim]
        attn_output = torch.matmul(attn_weight, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attn_output should be size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is "
                f"{attn_output.size()}"
            )
        
        # [Batch_size, Num_heads, Num_patches, Num_dim] -> [Batch_size, Num_pathces, Num_heads, Head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # [Batch_size, Num_patches, Num_heads, Num_dim] -> [Batch_size, Num_patches, Num_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # [Batch_size, Num_patches, Embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weight

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, config):
        super().__init__()
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Intermediate_size]
        hidden_states = self.fc1(hidden_states)

        # Hidden_states [Batch_size, Num_patches, Intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        # [Batch_size, Num_patches, Intermediate_size] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [Batch_size, Num_patches, Embed_dim]
        residual = hidden_states

        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.layer_norm_1(hidden_states)

        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.self_attn(hidden_states=hidden_states)

        # [Batch_size, Num_patches, Embed_dim]
        hidden_states = residual + hidden_states

        # Residual [Batch_size, Num_patches, Embed_dim]
        residual = hidden_states

        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.layer_norm_2(hidden_states)

        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.mlp(hidden_states)

        # [Batch_size, Num_patches, Embed_dim] 
        hidden_states = residual + hidden_states

        return hidden_states
    
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config=config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embed: torch.Tensor) -> torch.Tensor:
        # Input_embeds: [Batch_size, Num_patches, Embed_dim]
        hidden_states = input_embed

        for encoder_layer in self.layers:
            # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Pixel_values: [Batch_size, Channels, Height, Width] -> [Batch_size, Num_pathces, Embed_dims]
            hidden_states = self.embedding(pixel_values)

            last_hidden_state = self.encoder(inputs_embeds=hidden_states)

            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state
        

class SiglipVisionModel:
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)