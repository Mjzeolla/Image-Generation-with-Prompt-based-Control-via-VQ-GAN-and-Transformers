WORKDIR=work/vqvae/face-vqvae2
mkdir $WORKDIR -p
python train_vqvae.py -save \
  --log_interval=500 \
  --n_updates=10000\
  --dataset=face \
  --work_dir=$WORKDIR \
  --data_dir=data \
  --n_embeddings=64 \
  --embedding_dim=4 \
  --scaling_rates 4