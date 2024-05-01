import utils
import data
model = utils.load_module("work/results/vqvae.pt")
d = data.ImageNet("work", device="cuda")
es = utils.encode_latent(model, d[0:16][0])
utils.decode_latent(model, es, (64,64))