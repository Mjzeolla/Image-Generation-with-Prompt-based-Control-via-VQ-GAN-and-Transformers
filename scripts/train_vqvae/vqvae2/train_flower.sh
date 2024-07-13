WORKDIR=work/vqvae/flower-vqvae2
mkdir $WORKDIR -p
python train_vqvae.py -save \
  --log_interval=500 \
  --n_updates=30000\
  --data_dir=data \
  --dataset=flower \
  --work_dir=$WORKDIR\
  --n_embeddings=256 \
  --embedding_dim=4 \
  --scaling_rates 4