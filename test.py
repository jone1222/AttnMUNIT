"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import attention
import cv2


def save_attn_img(con_sty_attn_lst,attn_path):
    con_attn_lst, sty_attn_lst = con_sty_attn_lst[0], con_sty_attn_lst[1]

    #save content_img


    for idx in range(len(con_attn_lst)):

        attn_img = con_attn_lst[idx].cpu().data.numpy()
        path = os.path.join(attn_path, 'content_attn{:03d}.jpg'.format(idx))
        # Image.fromarray(attn_img.astype('uint8')).save(path)
        cv2.imwrite(path, cv2.cvtColor(attn_img,cv2.COLOR_BGR2RGB))

    for idx in range(len(sty_attn_lst)):
        attn_img = sty_attn_lst[idx].cpu().data.numpy()
        path = os.path.join(attn_path, 'style_attn{:03d}.jpg'.format(idx))
        # Image.fromarray(attn_img.astype('uint8')).save(path)
        cv2.imwrite(path, cv2.cvtColor(attn_img,cv2.COLOR_BGR2RGB))


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
style_decode = trainer.gen_a.decode if opts.a2b else trainer.gen_b.decode # decode function


num_channels = int(config['gen']['dim']*(2**config['gen']['n_downsample']))
hw_latent = int(config['crop_image_height']/(2**config['gen']['n_downsample']))

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
    style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

    # Start testing
    content, _style = encode(image)

    if opts.trainer == 'MUNIT':
        # style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
        style_rand = Variable(torch.randn(opts.num_style, num_channels, hw_latent, hw_latent).cuda())
        if opts.style != '':
            _content, style = style_encode(style_image)
        else:
            style = style_rand

        # recon image
        recon_input = style_decode(content, _style)


        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)

            #attention map of style
            gather_out_ab,distrib_out_ab,gather_out_ba,distrib_out_ba = \
                attention.visualize_attention_map(image, style_image, trainer.gen_a, trainer.gen_b, scale_factor=(2**config['gen']['n_downsample']))

            # gather_con_ab, gather_sty_ab = gather_out_ab[0], gather_out_ab[1]
            # for idx in range(len(gather_con_ab)):
            #     attn_img = gather_con_ab[idx].cpu().data.numpy()
            #     attn_path = os.path.join(opts.output_folder, 'attn_images/gather_con_ab/attn{:03d}.jpg'.format(idx))
            #     imwrite(attn_path,attn_img)
            #
            save_attn_img(gather_out_ab,os.path.join(opts.output_folder, 'attn_images/gather_ab/'))
            save_attn_img(gather_out_ba,os.path.join(opts.output_folder, 'attn_images/gather_ba/'))
            save_attn_img(distrib_out_ab,os.path.join(opts.output_folder, 'attn_images/distrib_ab/'))
            save_attn_img(distrib_out_ba,os.path.join(opts.output_folder, 'attn_images/distrib_ba/'))

            #recon input with random_style
            recon_style = style_decode(content,s)
            recon_style = (recon_style + 1) / 2.
            recon_path = os.path.join(opts.output_folder, 'recon_input_style{:03d}.jpg'.format(j))
            vutils.save_image(recon_style.data, recon_path, padding=0, normalize=True)



    elif opts.trainer == 'UNIT':
        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        path = os.path.join(opts.output_folder, 'output.jpg')
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
    else:
        pass

    if not opts.output_only:
        # also save input images
        vutils.save_image(image.data, os.path.join(opts.output_folder, 'input.jpg'), padding=0, normalize=True)
        # also save recon image
        vutils.save_image(recon_input.data, os.path.join(opts.output_folder, 'recon_input.jpg'), padding=0, normalize=True)

