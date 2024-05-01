WORKDIR=work/vqvae/mnist-vqvae2
mkdir $WORKDIR -p
python train_vqvae.py -save \
  --log_interval=1000 \
  --n_updates=5000\
  --dataset=mnist \
  --work_dir=$WORKDIR \
  --data_dir=data \
  --n_embeddings=8 \
  --embedding_dim=3 \
  --scaling_rates 2