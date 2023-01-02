# Faster RePaint
#### The implementation were forked and merged from:
```bash
https://github.com/NVlabs/denoising-diffusion-gan & https://github.com/andreas128/RePaint
```


## Faster RePaint fills a missing image part using denoising diffusion GANS

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 1" src="https://user-images.githubusercontent.com/11280511/150766080-9f3d7bc9-99f2-472e-9e5d-b6ed456340d1.gif"></td>
        <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 2" src="https://user-images.githubusercontent.com/11280511/150766125-adf5a3cb-17f2-432c-a8f6-ce0b97122819.gif"></td>
  </tr>
</table>


## Setup

### 1. Environment
```bash
pip install numpy torch blobfile tqdm pyYaml pillow  
```

### 2. Download models and data

```bash
pip install --upgrade gdown && bash ./download.sh
```
## Pretrained Checkpoints ##
We have released pretrained checkpoints on CIFAR-10 and CelebA HQ 256 at this 
[Google drive directory](https://drive.google.com/drive/folders/1UkzsI0SwBRstMYysRdR76C1XdSv5rQNz?usp=sharing).
Simply download the `saved_info` directory to the code directory. Use  `--epoch_id 550` for CelebA HQ 256 in the commands below.

### Evaluation
#### CelebA HQ 256 ####

We train Denoising Diffusion GANs on CelebA HQ 256 using 8 32-GB V100 GPUs. 
```
python3 train_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
--num_res_blocks 2 --batch_size 4 --num_epoch 800 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10  --num_process_per_node 8 --save_content
```


## Evaluation ##
After training, samples can be generated by calling ```test_ddgan.py```. We evaluate the models with single V100 GPU.
Below, we use `--epoch_id` to specify the checkpoint saved at a particular epoch.
Specifically, the script for generating samples on CelebA HQ is 
```
python3 test_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id $EPOCH
```


