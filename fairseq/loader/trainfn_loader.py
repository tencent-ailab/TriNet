from collections import Counter
from multiprocessing import Pool

import os
import shutil
import time
import random, copy
import yaml
import logging

import queue
from threading import Thread
import numpy as np
import torch

from omegaconf import MISSING, II, OmegaConf

from .utils import *
from fairseq import options, tasks
from fairseq.data import FileArkDataset
from fairseq.utils import import_user_module

from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger("fairseq.loader")

def register(parser):
    pass

def textloader(infile, task, dummy_args, cfg, rank):
    logger.info("Using threads to load files with queue size: {} on rank {}".format(cfg.task.queue_size, rank))
    q = queue.Queue(cfg.task.queue_size)

    thread = Thread(target = putThread, args = (q, TextGenerator, infile, task, cfg, rank))
    thread.daemon = True
    thread.start()
    while True:
        item = q.get()
        q.task_done()
        if item == None:
            break
        yield item
    thread.join()

def get_mask_precompute_kwargs(cfg):
    if cfg.precompute_mask_indices or cfg.tpu:
        assert (
            cfg.inferred_w2v_config is not None
        ), "inferred_w2v_config must be set"
        return OmegaConf.to_container(
            cfg.inferred_w2v_config, resolve=True, enum_to_str=True
        )
    else:
        return {}

def TextGenerator(infile, task, cfg, rank):
    lineid = 0
    for line in infile:
        lineid += 1
        #if lineid % cfg.distributed_training.distributed_world_size != rank:
        #    continue

        fname=line.strip()
        if len(fname) == 0:
            continue

        logger.info("Reading Text Data from File {} on rank {}".format(fname, rank))

        text_compression_level = getattr(
            TextCompressionLevel, str(cfg.task.text_compression_level)
        )

        #with open(cfg.dataset.conformer_cfg, 'r') as f:
        #    conformer_cfg = yaml.load(f, Loader=yaml.FullLoader)
        #loader_conf=conformer_cfg['loader_conf']

        ret_dataset = FileArkDataset(
            manifest_path=fname,
            sample_rate=cfg.task.sample_rate,
            max_sample_size=cfg.task.max_sample_size,
            min_sample_size=cfg.task.min_sample_size,
            pad=cfg.task.labels is not None or cfg.task.enable_padding,
            normalize=cfg.task.normalize,
            num_buckets=cfg.task.num_batch_buckets or int(cfg.task.tpu),
            compute_mask_indices=(cfg.task.precompute_mask_indices or cfg.task.tpu),
            text_compression_level=text_compression_level,
            **get_mask_precompute_kwargs(cfg.task),
        )

        logger.info("Loaded {} sentences in {}".format(len(ret_dataset.sizes), fname))
        yield ret_dataset, fname

    yield None, None

