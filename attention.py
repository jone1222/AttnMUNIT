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
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

# from utils import get_config
# config = get_config(opts.config)



class ContentStyleAttentionblock(nn.Module):
    def __init__(self, dim, norm='in'):
        super(StyleAttentionBlock, self).__init__()

        #content --> 256 * 4 * 4
        #style --> 256 * 1


    def forward(self, content, style):
        #Reshape content -> 256 * 16 // style --> 256 * 1
        content = content.view(content.size(0),content.size(1),-1)

        #1. expand style feature 256* 1 --> 256 * 16

        #2. element-wise multiplication with content feature --> 256 * 16  .. 256 * 16 --> 256 * 16

        #3. use fc layer 256 * 16 -> 256 * 1

        # softmax 256 * 1

        return

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


class ASquare(nn.Module):
    def __init__(self,in_channels,attention_dim,k = 2):
        super(ASquare,self).__init__()


        self.in_channels = in_channels
        self.attention_dim = attention_dim

        self.net_theta = nn.Conv2d(self.in_channels,self.attention_dim,kernel_size = 1,stride = 1,padding=0)
        self.net_pi = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, stride=1, padding=0)
        self.net_g = nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1, stride=1, padding=0)

        self.out_conv = nn.Conv2d(self.attention_dim, self.in_channels, kernel_size=1, stride = 1, padding = 0)

        self.K = k

        self.c_m = self.attention_dim
        self.c_n = self.attention_dim

        self.softmax = nn.Softmax()


    def forward(self,content,style):

        # content = content.view(content.size(0),content.size(1),-1)
        # style = style.view(style.size(0),style.size(1),-1)



        x = torch.cat((content,style),0)

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

        content_update, style_update = torch.split(output, 1, dim=0)

        return (content_update, style_update)



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
