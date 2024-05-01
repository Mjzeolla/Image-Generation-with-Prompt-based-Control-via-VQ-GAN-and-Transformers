1. `gen.py` defines core routine for image generation and training
2. `run.py` runs the training code and evaluation code.
3. `train.sh` and `eval.sh` set the parameter you need to use for training and evaluation
```
python run.py --dataset_type mnist \
       --work_dir work \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 300 \
       --n_embed 128 \
       --n_head 8 \
       --n_layer 4 \
       --pdropout 0.01 \
       --batch_size 32 \
       --learning_rate 6e-4 \
       --n_epoch 32 \
       --n_warmup 1000 \
       --downscale 2 \
       --row 5 \
       --num_beams 2 \
       --temperature 2 \
       --checkpoint work/model/final.pt
```
Important thing:
1. Set `work_dir` to somewhere you want to store the checkpoint and plot. On Turing cluster, this should be /work/your_wpi_username
2. If you want to train the model, set seed to zero. If you want to evaluate the model, just provide a checkpoint model file.
3. See `run.py` for help.