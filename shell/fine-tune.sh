fairseq-hydra-train \
    distributed_training.distributed_port=32678 \
    task.data=/root/lynncao/fairseq/librispeech-100h \
    distributed_training.distributed_world_size=1 \
    +optimization.update_freq='[8]' \
    model.w2v_path=/root/lynncao/fairseq/outputs/2021-08-31/17-54-03/checkpoints/checkpoint_best.pt \
    --config-dir ./examples/wav2vec/config/finetuning \
    --config-name base_100h
