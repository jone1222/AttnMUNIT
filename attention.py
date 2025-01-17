##########################
# Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
# Paper Link: https://arxiv.org/abs/1812.05252
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import cv2
from torchvision import utils

# from utils import get_config
# config = get_config(opts.config)

def visualize_attention_map(x_a, x_b, gen_a, gen_b, scale_factor=16):
    # for i in range(x_a.size(0)):

    c_a, s_a = gen_a.encode(x_a)
    c_b, s_b = gen_b.encode(x_b)

    x_ab = gen_b.decode(c_a, s_b)
    x_ba = gen_a.decode(c_b, s_a)

    # import pdb
    # pdb.set_trace()

    gather_out_ab, distrib_out_ab = get_attention_from_generator(x_a.squeeze(0),x_b.squeeze(0),gen_b, scale_factor=scale_factor)
    gather_out_ba, distrib_out_ba = get_attention_from_generator(x_a.squeeze(0),x_b.squeeze(0),gen_a, scale_factor=scale_factor)

    return gather_out_ab,distrib_out_ab,gather_out_ba,distrib_out_ba

def get_attention_from_generator(x_a,x_b,net_g, scale_factor):
    alpha_gathering, alpha_distribute = net_g.get_attention()

    gather_out = visualize_attention(x_a,x_b,alpha_gathering, scale_factor, shape_dim=5)
    distrib_out = visualize_attention(x_a,x_b,alpha_distribute, scale_factor, shape_dim=5)

    return gather_out, distrib_out

def visualize_attention(x_a, x_b, alpha, scale_factor, shape_dim=5):
    # output : numpy concatenated image 8*64

    if shape_dim == 5:
        # attention map shape --> batch, k , channels, h, w
        # 1 x 2 x 512 x 16 x 16
        # content_map, style_map = torch.split(alpha,1,dim=1)
        # content_map = content_map.squeeze(0)
        # style_map = style

        # batch, channels, h, w
        attention_map = alpha.squeeze(0)
    else:
        attention_map = alpha

    # up = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    # upsampled_map = up(attention_map)


    c_attention, s_attention = np.split(attention_map,2,axis=0)

    c_attn = cv2.resize(c_attention.reshape(16, 16, -1), (x_a.size(1), x_a.size(2)))
    c_attn_image_lst = draw_attention(x_a, c_attn)

    s_attn = cv2.resize(s_attention.reshape(16, 16, -1), (x_b.size(1), x_b.size(2)))
    s_attn_image_lst = draw_attention(x_b, s_attn)

    # pdb.set_trace()
    # upsampled_map = torch.nn.functional.interpolate(attention_map,scale_factor=scale_factor)
    # gaussian noise ?
    # output_map = upsampled_map
    attn_img_lst = [c_attn_image_lst, s_attn_image_lst]
    return attn_img_lst

def draw_attention(image,alpha_map,smooth = True):
    #channel-wise attention map
    alpha_map_lst = np.split(alpha_map, alpha_map.shape[2], axis=2)
    # pdb.set_trace()

    output_img_lst = []
    for alpha_map in alpha_map_lst:

        # alpha_map = alpha_map.squeeze(2)
        alpha_map = alpha_map - np.min(alpha_map)
        alpha_map = alpha_map / np.max(alpha_map)

        masked_img = np.multiply(((image+1)*128).permute(1,2,0),alpha_map)
        output_img_lst.append(masked_img)

        # cv2.imwrite('con_attn.jpg', masked_img)
        # utils.save_image(torch.from_numpy(masked_img),'cam.jpg')
    #
    # image = Image.fromarray(image.numpy())

    #alpha map : batch, size, asdf
    return output_img_lst

class StyleAttentionBlock(nn.Module):
    def __init__(self,dim,norm='in',is_first_level=True):
        super(StyleAttentionBlock,self).__init__()
        self.content_conv_block = self.build_conv_block(dim,norm)
        self.style_conv_block = self.build_conv_block(dim,norm,is_first_level)

    def build_conv_block(self,dim,norm,is_first_level=True):
        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d

        conv_block = []

        if is_first_level:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim), nn.ReLU()]
        else:
            conv_block += [nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1), norm_layer(dim * 2), nn.ReLU()]

        if is_first_level:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim), nn.ReLU()]
        else:
            conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1), norm_layer(dim * 2)]

        return  nn.Sequential(*conv_block)

    def forward(self,content,style):
        style_update = self.style_mask_net(style)
        style_mask = nn.functional.sigmoid(style_update)

        content_update = self.content_conv_block(content)

        content_update = content_update * style_mask

        content_output = content_update + content

        style_output = torch.cat((style_update,content_output),1)

        return content_output, style_output, content_update

class PATNBlock(nn.Module):
    def __init__(self, con_channels, style_channels, is_con_sty_concat=False):
        super(PATNBlock,self).__init__()

        self.con_channels = con_channels
        self.style_channels = style_channels

        self.con_stream = nn.Sequential(
            nn.Conv2d(in_channels = con_channels,out_channels=con_channels,kernel_size=3,padding=1),
            nn.InstanceNorm2d(num_features=con_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=con_channels, out_channels=con_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=con_channels)
        )

        self.style_stream = nn.Module()
        if is_con_sty_concat:
            sty_stream = nn.Sequential(
                nn.Conv2d(in_channels=con_channels + style_channels, out_channels=con_channels + style_channels,
                          kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=con_channels + style_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels= con_channels + style_channels, out_channels=con_channels, kernel_size=3,
                          padding=1),
                nn.InstanceNorm2d(num_features=con_channels)
            )
        else:
            sty_stream = nn.Sequential(
                nn.Conv2d(in_channels=style_channels, out_channels=style_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=con_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=style_channels, out_channels=style_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=con_channels)
            )
        self.style_stream = sty_stream

    def forward(self, con, sty):
        con_out = self.con_stream(con)
        sty_out = self.sty_stream(sty)
        torch.nn.functional.sigmoid(sty_out)

class ChannelAttention(nn.Module):
    def __init__(self, con_channels, style_channels, attn_channels, tau=0.5):
        super(ChannelAttention, self).__init__()

        #AttnGAN style attention

        #content --> b * c * w * h
        #style --> b * c * 1 * 1

        self.con_channels = con_channels
        self.attn_channels = attn_channels
        self.style_channels = style_channels

        self.net_q = nn.Conv2d(self.con_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_k = nn.Conv2d(self.style_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_v = nn.Conv2d(self.style_channels, self.style_channels, kernel_size=1, stride=1, padding=0)

        self.tau = tau
        # self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 1

    def forward(self, con, style):
        b, c, w, h = con.shape

        style_dim = style.size(1) #

        con_avg = F.adaptive_avg_pool2d(con,(1,1))
        con_q = self.net_q(con_avg)
        sty_k = self.net_k(style)
        sty_v = self.net_v(style)

        con_q = con_q.view(b, con_q.size(1), -1)
        sty_k_transpose = sty_k.view(b, sty_k.size(1), -1).permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        score = torch.bmm(con_q, sty_k_transpose) / math.sqrt(con_q.size(1))
        attn = F.softmax(score,dim=1)  # b * (h*w) * (h*w)

        # con_v = con_v.view(b, c, h * w)  # b * c * (h*w)


        out = torch.bmm(attn,sty_v.view(b,sty_v.size(1),-1))
        out = out.unsqueeze(3)

        per_channel_preserve_weight = F.sigmoid(out)

        #binarize
        per_channel_preserve_weight = torch.where(per_channel_preserve_weight > self.tau, torch.ones(1).cuda(), torch.zeros(1).cuda())
        # out = self.gamma * out + con

        return per_channel_preserve_weight

class ChannelAttention_FC(nn.Module):
    def __init__(self, con_channels, style_channels, tau=0.5 ):
        super(ConStyAttention, self).__init__()

        #AttnGAN style attention

        #content --> b * c * w * h
        #style --> b * c * 1 * 1

        self.con_channels = con_channels
        self.style_channels = style_channels
        self.tau = torch.Tensor(tau,requires_grad=False)

        self.fc_layer = nn.Conv2d(con_channels+style_channels, con_channels, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 1

    def forward(self, con, style):
        b, c, w, h = con.shape

        style_dim = style.size(1) #

        con_avg = F.adaptive_avg_pool2d(con,(1,1))

        out = self.fc_layer(torch.cat((con_avg,style),dim=1))

        per_channel_preserve_weight = F.sigmoid(out)

        #binarize
        per_channel_preserve_weight = torch.where(per_channel_preserve_weight > self.tau, 1, 0)
        # out = self.gamma * out + con
        attn = None

        return per_channel_preserve_weight, attn



class ConStyAttention(nn.Module):
    def __init__(self, con_channels, style_channels, attn_channels):
        super(ConStyAttention, self).__init__()

        #AttnGAN style attention

        #content --> b * c * w * h
        #style --> b * c * 1 * 1

        self.con_channels = con_channels
        self.attn_channels = attn_channels
        self.style_channels = style_channels

        self.net_sty = nn.Conv2d(self.style_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_con = nn.Conv2d(self.con_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_v = nn.Conv2d(self.style_channels, self.style_channels, kernel_size=1, stride=1, padding=0)

        # self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma = 1

    def forward(self, con, style):
        b,c,w,h = con.shape

        style_dim = style.size(1) #

        con_q = self.net_con(con)
        sty_k = self.net_sty(style)
        # con_v = self.net_v(style)

        con_q_transpose = con_q.view(b, con_q.size(1), -1).permute(0, 2, 1)
        sty_k = sty_k.view(b,-1,sty_k.size(2)*sty_k.size(3))
        # import pdb; pdb.set_trace()
        attn = nn.functional.softmax(torch.bmm(con_q_transpose, sty_k),dim=1)  # b * (h*w) * (h*w)

        # con_v = con_v.view(b, c, h * w)  # b * c * (h*w)

        style_v = self.net_v(style)
        out = torch.bmm(style_v.view(b,style_v.size(1),-1), attn.permute(0, 2, 1))
        out = out.view(b, c, w, h)

        out = self.gamma * out + con

        return out, attn

class SelfAttention(nn.Module):
    def __init__(self, in_channels, attn_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.attn_channels = attn_channels

        self.net_q = nn.Conv2d(self.in_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_k = nn.Conv2d(self.in_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)
        self.net_v = nn.Conv2d(self.in_channels, self.attn_channels, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.ones(1),requires_grad=False)

    def forward(self, input):
        #input ==> b x c x w x h

        b,c,w,h = input.shape

        q = self.net_q(input)
        k = self.net_k(input)
        v = self.net_v(input)

        v = v.view(b,c,h*w)
        k_transpose = k.view(b,c,h*w).permute(0,2,1)

        attn =  self.softmax(torch.bmm(v,k_transpose)) # b * c * c

        q = q.view(b,c,h*w)  # b * c * (h*w)

        out = torch.bmm(attn,q)
        out = out.view(b,c,w,h)

        out = self.gamma * out + input

        return out





class ASquare(nn.Module):
    def __init__(self,in_channels,attention_dim,k = 2):
        super(ASquare,self).__init__()


        self.in_channels = in_channels
        self.attention_dim = attention_dim

        self.net_theta = nn.Conv2d(self.in_channels,self.attention_dim,kernel_size = 1,stride = 1,padding=0)
        self.net_pi = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, stride=1, padding=0)
        self.net_g = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, stride=1, padding=0)

        self.out_conv = nn.Conv2d(self.attention_dim, self.in_channels, kernel_size=1, stride = 1, padding = 0)

        #if K = 1 --> work as Self-ATtention
        self.K = k

        self.c_m = self.attention_dim
        self.c_n = self.attention_dim

        self.softmax = nn.Softmax()


    def forward(self,content,style=None):

        # content = content.view(content.size(0),content.size(1),-1)
        # style = style.view(style.size(0),style.size(1),-1)

        if self.K > 1:
            x = torch.cat((content,style),0)
        else:
            x = content

        b, c, h, w = x.size()



        assert c == self.in_channels, 'input channel not equal!'
        # assert b//self.K == self.in_channels, 'input channel not equal!'
        feat_theta = self.net_theta(x)
        feat_pi = self.net_pi(x)
        feat_g = self.net_g(x)
        batch = int(b / self.K)

        # pdb.set_trace()
        tmp_theta = feat_theta.view(batch, self.K, self.c_m, h * w).permute(0, 2, 1, 3).contiguous().view(batch, self.c_m, self.K * h * w)
        tmp_pi = feat_pi.view(batch, self.K, self.c_n, h * w).permute(0, 2, 1, 3).contiguous().view(batch * self.c_n, self.K * h * w)
        tmp_g = feat_g.view(batch, self.K, self.c_n, h * w).permute(0, 1, 3, 2).contiguous().view(int(b * h * w), self.c_n)
        softmax_pi = self.softmax(tmp_pi).view(batch, self.c_n, self.K * h * w).permute(0, 2,1)  # batch, self.K*h*w, self.c_n
        softmax_g = self.softmax(tmp_g).view(batch, self.K * h * w, self.c_n).permute(0, 2, 1)  # batch, self.c_n , self.K*h*w
        tmpG = tmp_theta.matmul(softmax_pi)  # batch, self.c_m, self.c_n
        tmpZ = tmpG.matmul(softmax_g)  # batch, self.c_m, self.K*h*w
        tmpZ = tmpZ.view(batch, self.c_m, self.K, h * w).permute(0, 2, 1, 3).view(int(b), self.c_m, h, w)

        output = self.out_conv(tmpZ)

        if self.K > 1:
            content_update, style_update = torch.split(output, 1, dim=0)
        else:
            content_update = output

        alpha_gathering = softmax_pi.contiguous().view( batch, self.K, self.c_m, h, w )
        alpha_distribute = softmax_g.contiguous().view( batch, self.K, self.c_m, h, w )

        if self.K > 1:
            return content_update, style_update, alpha_gathering, alpha_distribute
        else:
            return content_update + content



        #
        # theta = self.net_theta(x)
        # pi = self.net_pi(x)
        # g = self.net_g(x)
        #
        # theta = theta.view(theta.size(0),theta.size(1),-1).transpose(1,2)
        # pi = pi.view(pi.size(0),pi.size(1),-1)
        # g = pi.view(g.size(0), g.size(1), -1)
        #
        # theta_mul_pi = nn.functional.softmax(theta @ pi)
        #
        # g_mul_softmax = (theta_mul_pi @ g)
        # g_mul_softmax = g_mul_softmax








class Net(nn.Module):
    """
    Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
    Based on code from https://github.com/Cyanogenoid/vqa-counting
    """

    def __init__(self):
        super(Net, self).__init__()
        # self.question_features = 1280
        # self.vision_features = config.output_features
        # self.hidden_features = 512
        # self.num_inter_head = 8
        # self.num_intra_head = 8
        # self.num_block = 2
        # self.visual_normalization = True

        self.style_features = 256
        self.content_features = 4096
        self.hidden_features = 512
        self.num_inter_head = 8
        self.num_intra_head = 8
        self.num_block = 1
        self.visual_normalization = True

        assert (self.hidden_features % self.num_inter_head == 0)
        assert (self.hidden_features % self.num_intra_head == 0)


        self.interIntraBlocks = SingleBlock(
            num_block=self.num_block,
            v_size=self.vision_features,
            q_size=self.question_features,
            output_size=self.hidden_features,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.0,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q):
        '''
        v: visual feature      [batch, 2048, num_obj]
        b: bounding box        [batch, 4, num_obj]
        q: question            [batch, max_q_len]
        '''
        # prepare v & q features
        v = v.transpose(1, 2).contiguous()
        b = b.transpose(1, 2).contiguous()
        q = self.text(q)  # [batch, max_len, 1280]
        if self.visual_normalization:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v)  # [batch, max_obj, 2048]
        v, q = self.interIntraBlocks(v, q)

        return v,q


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y) ** 2 + F.relu(x + y)


class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert (len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        # self.fusion = Fusion()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out


class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """

    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)

        self.interBlock = InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, q):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        for i in range(self.num_block):
            v, q = self.interBlock(v, q)
            v, q = self.intraBlock(v, q)
        return v, q


class MultiBlock(nn.Module):
    """
    Multi Block Inter-/Intra-modality
    """

    def __init__(self, num_block, v_size, q_size, output_size, num_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_block = num_block

        blocks = []
        blocks.append(InterModalityUpdate(v_size, q_size, output_size, num_head, drop))
        blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        for i in range(num_block - 1):
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, num_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        """
        for i in range(self.num_block):
            v, q = self.multi_blocks[i * 2 + 0](v, q)
            v, q = self.multi_blocks[i * 2 + 1](v, q)
        return v, q


class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0, downsampled_size = 256):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.downsampled_size = downsampled_size
        self.output_size = output_size
        self.num_head = num_head

        #self.v_size = [batch, channels, h*w] >> h*w = 4096
        self.downsample_content = []

        self.downsample_content += [nn.Conv2d(256, 256, 4, 2, 1,bias=True)]
        self.downsample_content += [nn.InstanceNorm2d(256)]
        self.downsample_content += [nn.ReLU(inplace=True)]
        self.downsample_content += [nn.Conv2d(256, 256, 4, 2, 1,bias=True)]
        self.downsample_content += [nn.InstanceNorm2d(256)]
        self.downsample_content += [nn.ReLU(inplace=True)]
        self.downsample_content = nn.Sequential(*self.downsample_content)

        self.map_style = []
        self.map_style += [nn.Conv2d(256, 256, 1, bias=True)]
        self.map_style += [nn.ReLU(inplace=True)]
        self.map_style = nn.Sequential(*self.map_style)

        # self.v_lin = nn.Linear(v_size, output_size * 3)
        # self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_lin = nn.Linear(self.downsampled_size, output_size * 3)
        self.q_lin = nn.Linear(self.downsampled_size, output_size * 3)


        # self.v_output = nn.Linear(output_size + v_size, output_size)
        # self.q_output = nn.Linear(output_size + q_size, output_size)

        self.v_output = nn.Linear(output_size + v_size, v_size)
        self.q_output = nn.Linear(output_size + q_size, q_size)


        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        """
        # batch_size, num_obj, _ = v.shape
        v_origin = v.view(v.size(0), v.size(1), -1)
        q_origin = q.view(q.size(0), q.size(1), -1)


        v = self.downsample_content(v)
        q = self.map_style(q)

        h, w = v.size(2), v.size(3)
        v = v.view(v.size(0), v.size(1), -1)
        q = q.view(q.size(0), q.size(1), -1)


        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply multi-head
        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)

        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  # [batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  # [batch, max_len, feat_size]
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (vq_slice @ qk_slice.transpose(1, 2)) / (
                              (self.output_size // self.num_head) ** 0.5)
            v2q = (qq_slice @ vk_slice.transpose(1, 2)) / (
                              (self.output_size // self.num_head) ** 0.5)
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2)  # [batch, num_obj, max_len]
            interMAF_v2q = F.softmax(v2q, dim=2)  # [batch, max_len, num_obj]
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            v_update = interMAF_q2v @ qv_slice if (i == 0) else torch.cat((v_update, interMAF_q2v @ qv_slice), dim=2)
            q_update = interMAF_v2q @ vv_slice if (i == 0) else torch.cat((q_update, interMAF_v2q @ vv_slice), dim=2)
        # update new feature
        # cat_v = torch.cat((v, v_update), dim=2)
        # cat_q = torch.cat((q, q_update), dim=2)
        cat_v = torch.cat((v_origin, v_update), dim=2)
        cat_q = torch.cat((q_origin, q_update), dim=2)

        updated_v = self.v_output(self.drop(cat_v))
        updated_q = self.q_output(self.drop(cat_q))
        return updated_v, updated_q


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """

    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        """

        # conditioned gating vector
        v_mean = v.sum(1) / v.shape[1]
        q_mean = q.sum(1) / q.shape[1]
        v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1)  # [batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1)  # [batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)

        # apply conditioned gate
        new_vq = (1 + q4v_gate) * v_q
        new_vk = (1 + q4v_gate) * v_k
        new_qq = (1 + v4q_gate) * q_q
        new_qk = (1 + v4q_gate) * q_k

        # apply multi-head
        vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
        vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
        qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  # [batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  # [batch, max_len, feat_size]
            # calculate attention
            v2v = (vq_slice @ vk_slice.transpose(1, 2)) / (
                              (self.output_size // self.num_head) ** 0.5)
            q2q = (qq_slice @ qk_slice.transpose(1, 2)) / (
                              (self.output_size // self.num_head) ** 0.5)
            dyIntraMAF_v2v = F.softmax(v2v, dim=2)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2)
            # calculate update input
            v_update = dyIntraMAF_v2v @ vv_slice if (i == 0) else torch.cat((v_update, dyIntraMAF_v2v @ vv_slice),
                                                                            dim=2)
            q_update = dyIntraMAF_q2q @ qv_slice if (i == 0) else torch.cat((q_update, dyIntraMAF_q2q @ qv_slice),
                                                                            dim=2)
        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))
        return updated_v, updated_q
