import torch
import argparse
import os
import sys
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from trainer import MUNIT_Trainer,AttnMUNIT_Trainer,UNIT_Trainer
from utils import get_model_list,get_config,pytorch03_to_pytorch04
from numpy import hstack, vstack

parser = argparse.ArgumentParser()
parser.add_argument('--configs', nargs='+', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoints', nargs='+', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
# parser.add_argument('--trainer', type=str, default='MUNIT', help="AttnMUNIT|MUNIT|UNIT")
parser.add_argument('--trainer', nargs='+', type=str, help="MUNIT AttnMUNIT UNIT")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

final_image_vstack = []

count = 0

for trainer_name in opts.trainer:
    # Load experiment setting
    config_name = opts.configs[count]
    print(config_name)
    config = get_config(config_name)


    opts.num_style = 1 if opts.style != '' else opts.num_style

    # Setup model and data loader
    if trainer_name == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif trainer_name == 'UNIT':
        trainer = UNIT_Trainer(config)
    elif trainer_name == 'AttnMUNIT':
        trainer = AttnMUNIT_Trainer(config)
    else:
        sys.exit("Only support AttnMUNIT|MUNIT|UNIT")
    # if opts.trainer == 'MUNIT':
    #     trainer = MUNIT_Trainer(config)
    # elif opts.trainer == 'UNIT':
    #     trainer = UNIT_Trainer(config)
    # elif opts.trainer == 'AttnMUNIT':
    #     trainer = AttnMUNIT_Trainer(config)
    # else:
    #     sys.exit("Only support AttnMUNIT|MUNIT|UNIT")

    try:
        state_dict = torch.load(opts.checkpoints[count])
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoints[count]), trainer_name)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])

    trainer.cuda()
    trainer.eval()

    encode = None
    style_encode = None
    decode = None
    style_decode = None
    if trainer_name in ['AttnMUNIT','MUNIT']:
        encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode  # encode function
        style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode  # encode function
        decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function
        style_decode = trainer.gen_a.decode if opts.a2b else trainer.gen_b.decode  # decode function

    if 'new_size' in config:
        new_size = config['new_size']
    else:
        if opts.a2b==1:
            new_size = config['new_size_a']
        else:
            new_size = config['new_size_b']
    style_dim = config['gen']['style_dim']


    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

        # Start testing
        content, _style = encode(image)

        if trainer_name in ['MUNIT','AttnMUNIT']:

            style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            # style_rand = Variable(torch.randn(opts.num_style, num_channels, hw_latent, hw_latent).cuda())
            if opts.style != '':
                _content, style = style_encode(style_image)
            else:
                style = style_rand

            # recon image
            recon_input = style_decode(content, _style)

            model_image_hstack = []

            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.

                path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
                grid = vutils.make_grid(outputs.data, padding=0, normalize=True)
                ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

                model_image_hstack.append(ndarr)

            model_image = hstack(model_image_hstack)
            final_image_vstack.append(model_image)

    count += 1

if len(final_image_vstack) > 0:
    final_image = vstack(final_image_vstack)
    im = Image.fromarray(final_image)
    im.save(os.path.join(opts.output_folder,'eval_result_multimodal.jpg'))