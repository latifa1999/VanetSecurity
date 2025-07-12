export CUDA_VISIBLE_DEVICES=3

model_name=NonStationaryTransformer

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Veremi/ \
  --model_id Veremi_cls \
  --model $model_name \
  --data veremi \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 2048 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3