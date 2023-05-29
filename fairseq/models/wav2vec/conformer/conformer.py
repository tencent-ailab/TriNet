from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.wav2vec.conformer.layer.transformer import ConformerEncoderLayer
from fairseq.models.wav2vec.conformer.layer.positionwise_feed_forward import PositionwiseFeedForward
from fairseq.models.wav2vec.conformer.layer.subsampling import LinearNoSubsampling
from fairseq.models.wav2vec.conformer.layer.subsampling import Conv2dSubsampling4
from fairseq.models.wav2vec.conformer.layer.subsampling import Conv2dSubsampling6
from fairseq.models.wav2vec.conformer.layer.subsampling import Conv2dSubsampling8
from fairseq.models.wav2vec.conformer.layer.positional_encoding import PositionalEncoding
from fairseq.models.wav2vec.conformer.layer.positional_encoding import RelPositionalEncoding
from fairseq.models.wav2vec.conformer.layer.positional_encoding import NoPositionalEncoding
from fairseq.models.wav2vec.conformer.layer.attention import MultiHeadedAttention
from fairseq.models.wav2vec.conformer.layer.attention import RelPositionMultiHeadedAttention
from fairseq.models.wav2vec.conformer.layer.convolution import ConvolutionModule
from fairseq.models.wav2vec.conformer.utils.common import get_activation
from fairseq.models.wav2vec.conformer.utils.mask import make_pad_mask
from fairseq.models.wav2vec.conformer.utils.mask import add_optional_chunk_mask


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        attention_heads: int = 4,
        attention_dim: int = 256,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        conv_subsample_in_ch: int = 1
    ):
        """
        Conformer from Wenet Implementation
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # subsampling
        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
            subsampling_args = (input_dim, attention_dim, dropout)
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.subsampling = subsampling_class(*subsampling_args)
        # positional embeding
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.pos_enc = pos_enc_class(attention_dim, positional_dropout_rate)
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        # attention layer
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "no_pos":
            selfattn_layer = MultiHeadedAttention
        else:
            selfattn_layer = RelPositionMultiHeadedAttention
        san_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
        )
        # feed-forward module in attention
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module in attention
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            attention_dim,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )
        # encoder blocks
        self.blocks = torch.nn.ModuleList([
            ConformerEncoderLayer(
                attention_dim,
                selfattn_layer(*san_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
        # output layer
        self.out_linear = nn.Linear(attention_dim, self.output_dim)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        output_embed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if not self.training:
            # set full chunk mode in validation
            decoding_chunk_size = -1
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        #print("conformer xs.shape:{}".format(" ".join([str(s) for s in xs.size()])))
        #print("conformer xs_lens.shape:{}".format(" ".join([str(s) for s in xs_lens.size()])))
        #print("conformer mask.shape:{}".format(" ".join([str(s) for s in masks.size()])))

        #xs, masks = self.subsampling(xs, masks)
        xs, pos_emb = self.pos_enc(xs)
        if xs.dtype == torch.float16:
            pos_emb = pos_emb.half()
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.blocks:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)

        #out = self.out_linear(xs)
        out=xs
        out_lens = masks.sum(dim=-1).view(-1)

        if not output_embed:
            return out, out_lens
        else:
            return out, out_lens, xs, masks

