python fairseq_cli/hydra_train.py \
task.data=librispeech-960h \
task.data_list=librispeech-960h/train.list \
checkpoint.save_dir=checkpoints/trinet_pretrain \
+optimization.update_freq='[1]' \
common.user_dir=examples/data2vec \
--config-dir examples/data2vec/config/audio/pretraining \
--config-name trinet_pretrain
