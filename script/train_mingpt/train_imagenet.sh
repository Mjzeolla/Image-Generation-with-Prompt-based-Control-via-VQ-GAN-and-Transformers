#!/usr/bin/sh
cp work/vqvae/imagenet_image_token.npy 'data/tiny-imagenet/image_token.npy'
python train_mingpt.py --dataset_type tiny-imagenet \
       --work_dir work/mingpt \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 3000 \
       --n_embed 512 \
       --n_head 8 \
       --n_layer 6 \
       --pdropout 0.1 \
       --batch_size 16 \
       --learning_rate 6e-4 \
       --n_epoch 256 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 1 \
       --data_dir=data \
       --vqvae_checkpoint=work/vqvae/results/imagenet-vqvae.pt \
       --unconditional
