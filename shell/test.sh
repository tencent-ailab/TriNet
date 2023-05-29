#!/bin/bash -x
test_set=(valid dev-other test-clean test-other)
word_score=0.0
sil_w=-0.2
lm_w=2.0

model=trinet_ft.pt
data_path=/apdcephfs/private_jiapengwang/all
model_path=/apdcephfs/private_jiapengwang/model/data2vec_finetune/400/checkpoint_best.pt

for test_set_i in ${test_set[@]}
do
    python3.7 examples/speech_recognition/new/infer.py \
    --config-dir examples/speech_recognition/new/conf \
    --config-name infer \
    task=audio_finetuning \
    task.data=${data_path}/librispeech-100h-ark-80/ \
    common.user_dir=examples/data2vec \
    task.labels=ltr \
    decoding.type=kenlm \
    decoding.lmweight=$lm_w \
    decoding.wordscore=$word_score \
    decoding.silweight=$sil_w \
    decoding.lexicon=${data_path}/librispeech-100h/lexicon \
    decoding.lmpath=${data_path}/4-gram.bin \
    decoding.unique_wer_file=True \
    dataset.gen_subset=$test_set \
    common_eval.path=${model_path} \
    decoding.beam=500 \
    distributed_training.distributed_world_size=1
done