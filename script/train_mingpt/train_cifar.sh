#!/usr/bin/sh
cp work/vqvae/cifar_image_token.npy data/cifar-10-batches-py/image_token.npy
python train_mingpt.py --dataset_type vqvae-cifar \
       --work_dir work/mingpt \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 3000 \
       --n_embed 512 \
       --n_head 8 \
       --n_layer 8 \
       --pdropout 0.0 \
       --batch_size 16 \
       --learning_rate 2e-4 \
       --n_epoch 32 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 1 \
       --data_dir=data \
       --vqvae_checkpoint=work/vqvae/results/cifar-vqvae.pt \
       --unconditional

        