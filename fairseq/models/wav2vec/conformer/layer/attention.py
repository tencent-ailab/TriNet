import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadAttnMemLayer(nn.Module):
    """
    Multi-head attention layer with or without consistent memory
    Args:
        - model_dim: dimension of query, key and value
        - head_num: number of attention heads
        - memory_num: number of memory vectors each head
    """
    def __init__(self, model_dim, head_num, memory_num=0, dropout=0.0):
        assert model_dim % head_num == 0
        super(MultiHeadAttnMemLayer, self).__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.memory_num = memory_num
        self.dim_each_head = model_dim // head_num
        if memory_num > 0:
            self.key_memory = nn.Parameter(
                    torch.zeros(head_num, memory_num, self.dim_each_head))
            nn.init.xavier_uniform_(self.key_memory)
            self.value_memory = nn.Parameter(
                    torch.zeros(head_num, memory_num, self.dim_each_head))
            nn.init.xavier_uniform_(self.value_memory)

        self.linear_query = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_key = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_value = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors
        Args:
            - key (FloatTensor): shape is [batch, key_len, model_dim]
            - value (FloatTensor): shape is [batch, key_len, model_dim]
            - query (FloatTensor): shape is [batch, query_len, model_dim]
            - mask (BooleanTensor): mask area which shouldn't be attended, if None,
                                    shape should be [batch, query_len, key_len]
        Returns:
            - out: output context vectors, [batch, seq_len, model_dim]
            - attn_vec: attention vectors on keys, [batch, head_num, seq_len, seq_len]
            - attn_mem: attention vectors on memorys, None or [batch, head_num, seq_len, memory_num]
        """
        b_k, l_k, d_k = key.size()
        b_v, l_v, d_v = value.size()
        b_q, l_q, d_q = query.size()
        assert b_k == b_v and b_k == b_q
        assert l_k == l_v
        if mask is not None:
            b_m, l_m, d_m = mask.size()
            assert b_m == b_q and l_m == l_q and d_m == l_k
        head_num = self.head_num
        memory_num = self.memory_num
        dim_each_head = self.dim_each_head
        # fold each head into batch dimension
        def shape_projection(x):
            b, l, d = x.size()
            return x.view(b, l, head_num, dim_each_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b * head_num, l, dim_each_head)
        # unfold each head into last dimension
        def unshape_projection(x):
            bh, l, d = x.size()
            b = bh // head_num
            return x.view(b, head_num, l, dim_each_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b, l, head_num * dim_each_head)

        key_up = shape_projection(self.linear_key(key))
        value_up = shape_projection(self.linear_value(value))
        if memory_num > 0:
            expand_key_mem = self.key_memory.repeat(b_k, 1, 1)
            key_up = torch.cat([key_up, expand_key_mem], dim=1)
            expand_value_mem = self.value_memory.repeat(b_v, 1, 1)
            value_up = torch.cat([value_up, expand_value_mem], dim=1)
        query_up = shape_projection(self.linear_query(query))
        # [batch * head_num, query_len, key_len]
        score = torch.bmm(query_up, key_up.transpose(1, 2))
        score = score / math.sqrt(self.dim_each_head)
        bh, ql, kl = score.size()
        b = bh // self.head_num
        if mask is not None:
            score = score.view(b, self.head_num, ql, kl)
            if memory_num > 0:
                pad_mem_mask = mask.new_zeros((b, ql, memory_num))
                mask = torch.cat([mask, pad_mem_mask], dim=2)
            # expand along attention head dimension
            mask = mask.unsqueeze(1).expand_as(score)
            score = score.masked_fill(mask, -float('inf')) \
                         .view(bh, ql, kl)
        attn = self.sm(score)
        attn = self.dropout(attn)
        # [batch, l, head_num * dim_each_head]
        out = self.linear_out(unshape_projection(torch.bmm(attn, value_up)))
        attn_vec = attn.view(b, self.head_num, ql, kl)
        attn_mem = None
        if memory_num > 0:
            attn_mem = attn_vec[:, :, :, -memory_num:]
            attn_vec = attn_vec[:, :, :, 0:-memory_num]
        return out, attn_vec, attn_mem


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor,
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            #print("att mask.shape:{}".format(" ".join([str(s) for s in mask.size()])))
            #print("att score.shape:{}".format(" ".join([str(s) for s in scores.size()])))

            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor = torch.empty(0),) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
