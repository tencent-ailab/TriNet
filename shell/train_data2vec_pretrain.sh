CURDIR=$(pwd)
python fairseq_cli/hydra_train.py \
task.data=${CURDIR}/librispeech-960h \
task.data_list=${CURDIR}/librispeech-960h/train.list \
checkpoint.save_dir=${CURDIR}/checkpoints/data2vec_pretrain \
+optimization.update_freq='[1]' \
common.user_dir=${CURDIR}/examples/data2vec \
--config-dir ${CURDIR}/examples/data2vec/config/audio/pretraining \
--config-name data2vec_pretrain
