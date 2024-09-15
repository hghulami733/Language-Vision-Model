import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self) -> None:
        self.key_cache = List[torch.Tensor] = []
        self.value_cache = List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_szie, Num_heads_kv, Seq_len, Head_dim]
            return self.key_cache[0].shape[-2]
        
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV_Cache of this layer, let's create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # ... Otherwise we concatenate the new key with the existing ones.
            # Each tensor has shape: [Batch_size, Num_heads_kv, Seq_len, Head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... And then we return all the existing key + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    def __init(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attnetion_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout= attnetion_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 ignore_index=-100,
                 image_token_index=256000,
                 vocab_size=257152,
                 projection_dim=2048,
                 hidden_size=2048,
                 pad_token_id=None,
                 **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w.whilst Gemma is (x * x).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivelent ti:
        # y = self.gate_proj(x) # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Intermediate_size]
        # y = self.torch.gelu(y, approximate='tanh') # [Batch_size, Seq_len, Intermediate_size]
        # j = self.up_proj(x) # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Intermediate_size]
        # z = y * j # [Batch_size, Seq_len, Intermediate_size] -> [Batch_size, Seq_len, Hidden_size]
        # z = self.down_proj(z) # [Batch_size, Seq_len, Intermediate_size] -> [Batch_size, Seq_len, Hidden_size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim # It is set to the head_dim
        selfmax_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base ^ (2i / dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad
    def forward(self, x, position_ids, seq_len=None):
        # x : [bs, num_attention_heads, seq_lwn, head_size]
        self.inv_freq.to(x.device)

        # Copy the inv_freq tensor for batch in the sequence
        # Inv_freq_expanded: [Batch_size, Head_dim // 2, 1]
        inv_freq_expanded = self.inv_fre[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # Position_ids_expanded: [Batch_size, 1, Seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device_type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position(which is the argument of the sin and cos functions)
            # Freqs: [Batch_size, Head_dim // 2, 1] @ [Batch_size, 1, Seq_len] --> [Batch_size, Seq_len, Head_dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # emb: [Batch_size, Seq_len, Head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)

            # Cos, Sin: [Batch_size, Seq_len, Head_dim]
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimention
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimention
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimention
    
    # Apply the formula (34) of the Rotary Positional Encoding paper
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()      
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num, * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len , _ = hidden_states.size() # [Batch_size, Seq_len, Hidden_size]

        # [Batch_size, Seq_len, Num_heads_q * Head_dim]
        query_states = self.q_proj(hidden_states)

        # [Batch_size, Seq_len, Num_heads_kv * Head_dim]
        key_states = self.k_proj(hidden_states)

        # [Batch_size, Seq_len, Num_heads_q * Head_dim]
        value_states = self.v_proj(hidden_states)

        # [Batch_size, Num_heads_q, Seq_len, Head_dim]
        query_states = query_states.veiw(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [Batch_size, Num_heads_kb, Seq_len, Head_dim]
        key_states = key_states.veiw(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_size, Num_heads_kv, Seq_len, Head_dim]
        value_states = value_states.veiw(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_size, Seq_len, Head_dim] -> [Batch_size, Seq_len, Head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # [Batch_size, Num_heads_q, Seq_len, Head_dim], # [Batch_size, Seq_len, Num_heads_kv, Seq_len, Head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Perfoem the calculation as usual, Q * K^T / sqrt(head_dim). shape: # [Batch_size, Num_heads_q, Seq_len_q, Seq_len_kv]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [Batch_size, Num_heads_q, Seq_len_q, Seq_len_kv]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Multiply by the values: [Batch_size, Num_heads_q, Seq_len_q, Seq_len_kv] x [Batch_size, Seq_len_q, Seq_len_kv, Head_dim] -> [Batch_size, Num_heads_q, Seq_len_q, Seq_len_kv]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)} but is "
                f"{attn_output.size()}"
            )
        
        # Make sure the sequence length is the second dimension. [Batch_size, Num_heads_q, Seq_len_q, Head_dim] -> [Batch_size, Seq_len_q, Num_heads, Seq_len_kv]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Concatenate all the heads together. [Batch_size, Seq_len_q,Num_heads_q, Head_dim] -> [Batch_size, Seq_len_q, Num_heads_q * Head_dim]
        attn_output = attn_output.veiw(bsz, q_len, -1)

        # Multiply by W_o. [Batch_size, Seq_len_q, Hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = config.hidden_size

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = residual + hidden_states

        # [Batch_size, Seq_len, Hidden_size]
        residual = hidden_states

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.mlp(hidden_states)

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states
    

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()   
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.Module(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        
        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = inputs_embeds

        # [Batch_size, Seq_len, Hidden_size]
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_size, Seq_len, Hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

        # [Batch_size, Seq_len, Hidden_size]
        hidden_states = self.norm(hidden_states)

        # [Batch_size, Seq_len, Hidden_size]
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bais=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weights = self.model.embed_tokens.weight

    def froward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                kv_cache: Optional[KVCache] = None,) -> Tuple:
        # Input_embeds: [Batch_size, Seq_len, Hidden_size] 
        # Output: [Batch_size, Seq_len, Hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModelProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_size, Num_patches, Embed_dim] ->[Batch_size, Num_patches, Projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForCoditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModelProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
            self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask:torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds, inputs_embeds.device

        # Shape: [Batch_size, Seq_len, Hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Shape: [Batch_size, Seq_len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & ( input_ids != self.pad_token_id)

        # Shape: [Batch_size, Seq_len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index

        # Shape: [Batch_size, Seq_len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the mask to the embedding dimension otherwise we can not use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        # Inser image embeddings. we can not use torch.wherer because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE the attention mask
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            """
            Do not mask any token, because we are in the profill phase
            This only works when we have no padding
            """
            causal_mask = torch.full(

                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we do not need to mask anything, since each query should be able to attend all previous
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        
        # Add the head dimension
        # [Batch_size, Q_len, Kv_len] -> [Batch_size, Num_heads_q, Q_len, Kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
    
    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_size, seq_len, Hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embed_dim]
        selected_iamge_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Hidden_size]
        image_features = self.multi_model_projector(selected_iamge_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_feature(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids= position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
    
