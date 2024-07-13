#!/usr/bin/sh
cp work/vqvae/face_image_token.npy data/face/image_token.npy
python train_mingpt.py --dataset_type vqvae-face \
       --work_dir work/mingpt \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 3000 \
       --n_embed 256 \
       --n_head 8 \
       --n_layer 4 \
       --pdropout 0.0 \
       --batch_size 16 \
       --learning_rate 2e-4 \
       --n_epoch 64 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 1 \
       --data_dir=data \
       --vqvae_checkpoint=work/vqvae/results/face-vqvae.pt \
       --unconditional