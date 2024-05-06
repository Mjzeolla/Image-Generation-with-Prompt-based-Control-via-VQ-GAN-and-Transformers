[embed][Image-Generation-with-Prompt-based-Control-via-VQ-GAN-and-Transformers](https://github.com/Mjzeolla/Image-Generation-with-Prompt-based-Control-via-VQ-GAN-and-Transformers/blob/main/Image-Generation-with-Prompt-based-Control-via-VQ-GAN-and-Transformers.pdf)[/embed]

To run conditional image generation, you need:
1. Download the text-image pair from the `https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view`
2. Unzip the file into `data/` and rename that directory to be `flower`. You should get a `data/flower/text_c10` folder.
3. Run `sh script/train_vqvae/unet_vqvae/train_flower.sh` to generate image tokens and vqvae encoder.
4. Open `train_flower_unet_conditional.ipynb` to run the generation code.

0. Create necessary folder
Create `work/vqvae` and `work/mingpt` folder in this folder. They are used to store temporary results.
1. Dataset preparation.
We currently have multiple datasets. MINST/Fashion-MNIST/CIFAR doesn't require manual downloading, where face and flower dataset require so.
Create a `data/` directory. Download flower dataset from `https://www.kaggle.com/datasets/yousefmohamed20/oxford-102-flower-dataset` and face dataset from `https://www.kaggle.com/datasets/splcher/animefacedataset`. Unzip them into `data/` folder. Ensure that the folder name is `102 flower` and `face`.2. VQVAE encoding
Run `sh script/train_vqvae/train_{datasetname}.sh` to generate encoding for the dataset. For exampple, `train_mnist` will generate encoding for mnist dataset. The result is stored in `work/vqvae`
3. Image generation
Run `sh script/train_mingpt/train_{datasetname}.sh` to run the generation pipeline. If you do latent space generation, you need to get VQVAE encoding first!

The model is defined in `mingpt` and `vqvae`. The training code is defined in `train_vqvae.py` and `training_mingpt.py`. If you want to change some configuration, then just copy the script in the script folder and change the command line parameters.

I suggest using IPython+vscode to develop your codes. Vscode supports shift+enter to evaluate codes block by block (like jupyter notebook). You can also use jupyter notebook.
