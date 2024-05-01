#!/usr/bin/sh
VQVAE_WORKDIR=work/vqvae/flower-unet-vqvae
ln -s "$PWD/$VQVAE_WORKDIR/flower_image_token.npy" 'data/102 flower/image_token.npy' -f
me=$(basename "$0")
WORKDIR="work/mingpt/$me"
python train_mingpt.py --dataset_type vqvae-flower \
       --work_dir $WORKDIR \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 3000 \
       --n_embed 256 \
       --n_head 8 \
       --n_layer 4 \
       --pdropout 0.01 \
       --batch_size 16 \
       --learning_rate 6e-4 \
       --n_epoch 256 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 1 \
       --data_dir=data \
       --vqvae_checkpoint="$VQVAE_WORKDIR/results/flower-vqvae.pt"
