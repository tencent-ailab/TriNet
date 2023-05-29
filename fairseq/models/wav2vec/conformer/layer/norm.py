import torch
import torch.nn as nn
from torch.nn import Parameter, init


class MaskBatchNorm(nn.Module):
    """
    BatchNorm layer where statistics are collected from masked frames
    Args:
        - feat_shape: shape for running_mean and running_var
        - eps: epsilon to avoid zero division
        - momentum: momentum from moving average on running_mean and running_var
        - affine: whether affine the normalized distribution
        - track_running_stats: whether to track running statistics
    Note:
        running statistics would not be synchronized in BMUF training, we believe
        statistics collected by one worker is sufficient and only store statistics
        from one worker (e.g. MASTER node)
    """
    def __init__(self, feat_shape, eps=1e-8, momentum=0.99,
                 affine=True, track_running_stats=True):
        super(MaskBatchNorm, self).__init__()
        self.feat_shape = feat_shape
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.ones(feat_shape))
            self.bias = Parameter(torch.zeros(feat_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(feat_shape))
            self.register_buffer("running_var", torch.ones(feat_shape))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs, mask, reduce_dims):
        assert len(mask.shape) == 1
        assert inputs.shape[0] == mask.shape[0]
        masked_input = inputs[mask]
        # train mode
        if self.training:
            # collect statistics from masked frames
            sample_mean = torch.mean(masked_input, reduce_dims, keepdim=True)
            sample_var = torch.var(masked_input, reduce_dims, keepdim=True)
            if self.track_running_stats:
                average_factor = self.momentum
                self.running_mean.data = self.running_mean.data * average_factor + \
                                            sample_mean * (1 - average_factor)
                self.running_var.data = self.running_var.data * average_factor + \
                                            sample_var * (1 - average_factor)
            normed_input = (inputs - sample_mean) / torch.sqrt(sample_var + self.eps)
        # eval mode with running_stats
        elif self.track_running_stats:
            normed_input = (inputs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        # eval mode without running_stats
        else:
            sample_mean = torch.mean(masked_input, reduce_dims, keepdim=True)
            sample_var = torch.var(masked_input, reduce_dims, keepdim=True)
            normed_input = (inputs - sample_mean) / torch.sqrt(sample_var + self.eps)
        # affine
        if self.affine:
            normed_input = normed_input * self.weight + self.bias
        return normed_input


class VarLenInstanceNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-8, affine=False):
        super(VarLenInstanceNorm2d, self).__init__()
        self.channels = channels
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = Parameter(torch.ones(channels))
            self.beta = Parameter(torch.zeros(channels))

    def forward(self, inputs, seq_len, seq_mask):
        assert inputs.dim() == 4
        batch_size, in_channel, time_step, feat_dim = inputs.size()
        assert in_channel == self.channels
        num_bins = seq_len.view(-1, 1, 1, 1) * feat_dim
        # mask before stat
        masked_inputs = inputs * seq_mask.unsqueeze(1).unsqueeze(3)
        mean = masked_inputs.sum(dim=[1, 2], keepdim=True) / num_bins
        var = (masked_inputs - mean) ** 2
        var = var * seq_mask.unsqueeze(1).unsqueeze(3)
        var = torch.sum(var, dim=[1, 2], keepdim=True) / num_bins
        normed_inputs = (masked_inputs - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            normed_inputs = normed_inputs * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return normed_inputs


def len2flatMask(seq_len, max_len, batch_first=False):
    """
    compute flat mask of valid frames from sequence length
    """
    assert seq_len.dim() == 1
    batch_size = seq_len.size(0)
    mask = (torch.arange(0, max_len)
            .type_as(seq_len)
            .repeat(batch_size, 1)
            .lt(seq_len.unsqueeze(1)))
    if not batch_first:
        mask = mask.transpose(0, 1).contiguous()
    return mask.view(-1)

