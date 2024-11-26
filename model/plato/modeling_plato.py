import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.models.bert.modeling_bert import BertPooler
from transformers.file_utils import ModelOutput
# from transformers.modeling_outputs import ModelOutput


import config as user_config
from model.plato.configuration_plato import PlatoConfig

@dataclass
class PlatoModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    response_selection_scores: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None


class Embedder(nn.Module):
    """
    Composite embedding layer.
    ref: Pytorch-PLATO
    """

    def __init__(self, config):
        super(Embedder, self).__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.pos_embedding.weight.requires_grad = config.pos_trainable
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.turn_embedding = nn.Embedding(config.turn_vocab_size, config.hidden_size)
        self.dropout_layer = nn.Dropout(p=config.hidden_dropout_prob)

        # follow the default xavier_uniform initializer in paddle version
        # otherwise, there are bugs for dec_probs computation in weight typing setting
        # default norm initializer in nn.Embedding in pytorch, which samples larger values
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        return

    def forward(self, token_inp, pos_inp, type_inp, turn_inp):
        embed = self.token_embedding(token_inp) + \
            self.pos_embedding(pos_inp) + \
            self.type_embedding(type_inp) + \
            self.turn_embedding(turn_inp)
        embed = self.dropout_layer(embed)
        return embed


class MultiheadAttention(nn.Module):
    """
    Multi head attention layer.
    """

    def __init__(self, config):
        assert config.hidden_size % config.num_attention_heads == 0
        super(MultiheadAttention, self).__init__()
        self.config = config

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.scale = self.attention_head_size ** -0.5
        self.linear_qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_layer = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        return

    def _split_heads(self, x, is_key=False):
        x = x.reshape(x.size(0), x.size(1), self.config.num_attention_heads, self.attention_head_size)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), self.config.hidden_size)
        return x

    def _attn(self, query, key, value, mask):
        # shape: [batch_size, num_head, seq_len, seq_len]
        scores = torch.matmul(query, key)
        scores = scores * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.config.num_attention_heads, 1, 1)
            scores.masked_fill_(mask.bool(), -1e10)  # scores = (1 - mask) * scores + mask * (-1e10)

        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)

        if mask is not None:
            '''
            mask: [batch size, num_attention_heads, seq_len, seq_len]
            mask后两维(seq_len, seq_len)矩阵来看，其中有的行可能都是true(1)，对应句子中<pad>位看的行
            导致softmax后该行的每个位置的attn prob都为1/n而非0，所以此处需重置为0

            >>> F.softmax([-1e10, -100, -100])
            >>> [0.00, 0.50, 0.50]
            >>> F.softmax([-1e10, -1e10, -1e10])
            >>> [0.33, 0.33, 0.33]
            ==> [0.00, 0.00, 0.00]
            '''
            attn.masked_fill_(mask.bool(), 0.)  # attn = (1 - mask) * attn

        out = torch.matmul(attn, value)
        return out

    def forward(self, inp, mask=None, cache=None):
        """ Forward process of self attention. """
        # shape: [batch_size, seq_len, 3 * hidden_size]
        qkv = self.linear_qkv(inp)
        query, key, value = torch.split(qkv, self.config.hidden_size, dim=2)

        # shape: [batch_size, num_head, seq_len, head_dim]
        query = self._split_heads(query)
        # shape: [batch_size, num_head, head_dim, seq_len]
        key = self._split_heads(key, is_key=True)
        # shape: [batch_size, num_head, seq_len, head_dim]
        value = self._split_heads(value)

        if cache is not None:
            if "key" in cache and "value" in cache:
                key = torch.cat([cache["key"], key], dim=3)
                value = torch.cat([cache["value"], value], dim=2)
            cache["key"] = key
            cache["value"] = value

        out = self._attn(query, key, value, mask)
        out = self._merge_heads(out)
        out = self.linear_out(out)
        return out


class FeedForward(nn.Module):
    """
    Positional feed forward layer.
    """

    def __init__(self, hidden_size, inner_dim, dropout):
        super(FeedForward, self).__init__()

        self.hidden_size = hidden_size
        self.inner_dim = inner_dim
        self.linear_hidden = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU()
        )
        self.linear_out = nn.Linear(inner_dim, hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, x):
        out = self.linear_hidden(x)
        out = self.dropout_layer(out)
        out = self.linear_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block module.
    """

    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config

        self.attn = MultiheadAttention(config)
        self.attn_norm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                      eps=1e-6,
                                      elementwise_affine=True)
        self.ff = FeedForward(hidden_size=config.hidden_size,
                              inner_dim=4 * config.hidden_size,
                              dropout=config.hidden_dropout_prob)
        self.ff_norm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                    eps=1e-6,
                                    elementwise_affine=True)
        self.dropout_layer = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, inp, mask=None, cache=None):
        """
        Forward process on one transformer layer.

        @param : x
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : memory
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : mask

        @param : cache
        """
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


class PlatoModel(nn.Module):
    def __init__(self, config, add_pooling_layer=True):
        super(PlatoModel, self).__init__()
        self.config = config

        self.embedder = Embedder(config)
        self.embed_layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                             eps=1e-6,
                                             elementwise_affine=True)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self._create_parameters()

    def _create_parameters(self):
        """ Create model's paramters. """
        if self.config.num_latent > 0:
            self.mask_embed = nn.Parameter(torch.Tensor(1, 1, self.config.hidden_size))
            self.latent_embeddings = nn.Parameter(torch.Tensor(self.config.num_latent, self.config.hidden_size))
            nn.init.normal_(self.mask_embed, std=0.02)
            nn.init.normal_(self.latent_embeddings, std=0.02)

        self._dtype = 'float32'
        sequence_mask = np.tri(self.config.max_position_embeddings, self.config.max_position_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask).to('cuda')
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        """
        Create attention mask.
        创建从序列形式到矩阵形式的mask：[batch_size, max_seq_len， 1] -> [batch_size, max_seq_len, max_seq_len]
        mask除了要考虑attention mask（自回归），还需要考虑pad的mask（自回归和双向）
        注：
        1. 一个句子中的非<pad>词看整个句子，该句中只有<pad>词才被mask
        2. 一个句子中的<pad>词看整个句子，该句的所有词都应该被mask

        @param : input_mask
        @type : Variable(shape: [batch_size, max_seq_len])

        @param : auto_regressive
        @type : bool
        """
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            # 拼接上 句首位置([M]/z)的mask
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            # 将tgt端的<pad> mask和自回归attention mask融合
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            turn_ids=None,
            role_ids=None,
            return_dict=False):
        embed = self.embedder(token_inp=input_ids,
                              pos_inp=position_ids,
                              type_inp=role_ids,
                              turn_inp=turn_ids)
        embed = self.embed_layer_norm(embed)
        all_hidden_states = ()
        for layer in self.layers:
            mask = self._create_mask(attention_mask, auto_regressive=False, append_head=False)
            embed = layer(inp=embed,
                          mask=mask,
                          cache=None)
            all_hidden_states = all_hidden_states + (embed,)
        last_hidden_state = embed
        response_selection_scores = self.discriminator(last_hidden_state)
        pooler_output = self.pooler(last_hidden_state) if self.pooler is not None else None

        if not return_dict:
            return all_hidden_states, last_hidden_state, pooler_output, response_selection_scores

        return PlatoModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            pooler_output=pooler_output,
            response_selection_scores=response_selection_scores,
        )


if __name__ == "__main__":
    config = PlatoConfig.from_json_file(user_config.plato_config_file)
    bert = PlatoModel(config)
