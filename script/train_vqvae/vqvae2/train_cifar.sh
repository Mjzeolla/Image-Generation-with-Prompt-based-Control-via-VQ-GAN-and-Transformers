WORKDIR=work/vqvae/cifar-vqvae2
mkdir $WORKDIR -p
python train_vqvae.py -save \
  --log_interval=500 \
  --n_updates=10000\
  --dataset=cifar \
  --work_dir=$WORKDIR  \
  --data_dir=data \
  --n_embeddings=16 \
  --scaling_rates 2