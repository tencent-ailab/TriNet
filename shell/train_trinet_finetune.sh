python fairseq_cli/hydra_train_ft.py \
task.data=librispeech-100h \
checkpoint.save_dir=checkpoints \
+optimization.update_freq='[1]' \
common.user_dir=examples/data2vec \
model.w2v_path=checkpoints/trinet_pretrain/checkpoint_best.pt \
--config-dir examples/data2vec/config/audio/finetuning \
--config-name base_100h