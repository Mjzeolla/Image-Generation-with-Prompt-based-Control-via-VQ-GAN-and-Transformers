#!/usr/bin/sh
cp work/vqvae/mnist_image_token.npy data/MNIST/image_token.npy
python train_mingpt.py --dataset_type vqvae-mnist \
       --work_dir work/mingpt \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 1000 \
       --n_embed 128 \
       --n_head 8 \
       --n_layer 4 \
       --pdropout 0.0 \
       --batch_size 32 \
       --learning_rate 6e-4 \
       --n_epoch 32 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 2 \
       --seed 0 \
       --data_dir=data \
       --vqvae_checkpoint=work/vqvae/results/mnist-vqvae.pt \
       --unconditional