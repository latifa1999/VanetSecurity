export CUDA_VISIBLE_DEVICES=2

model_name=Informer

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Veremi/ \
  --model_id Veremi_clss \
  --model $model_name \
  --data veremi \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10