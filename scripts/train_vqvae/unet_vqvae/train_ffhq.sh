WORKDIR=work/vqvae/ffhq-unet-vqvae
mkdir $WORKDIR -p
python train_vqvae.py -save \
  --vqvae_type unet \
  --log_interval=500 \
  --n_hiddens=32\
  --n_residual_layers=1\
  --n_updates=30000\
  --data_dir=data \
  --dataset=ffhq \
  --work_dir=$WORKDIR\
  --n_embeddings=256 \
  --embedding_dim=4 \
  --ch_mult 1 2 4