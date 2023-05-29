CURDIR=$(pwd)
python fairseq_cli/hydra_train_ft.py \
task.data=${CURDIR}/librispeech-100h \
checkpoint.save_dir=${CURDIR}/checkpoints \
+optimization.update_freq='[1]' \
common.user_dir=${CURDIR}/examples/data2vec \
model.w2v_path=${CURDIR}/checkpoints/trinet_pretrain/checkpoint350.pt \
--config-dir ${CURDIR}/examples/data2vec/config/audio/finetuning \
--config-name base_100h
