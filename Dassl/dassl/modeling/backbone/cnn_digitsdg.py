import torch.nn as nn
from torch.nn import functional as F

from dassl.utils import init_network_weights

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

from .styleaugment.styleaug.ghiasi import Ghiasi
from .styleaugment.styleaug.stylePredictor import StylePredictor
import numpy as np
import sys
from os.path import join, dirname
import torch
from torchvision.transforms import Normalize
import random
from dassl.modeling.ops import MixStyle

class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class ConvNetMix(Backbone):
    """CNN + MixStyle."""

    def __init__(self, c_hidden=64, mixstyle_layers=[]):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.mixstyle_layers = mixstyle_layers
        print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        if 'conv1' in self.mixstyle_layers:
            x = self.mixstyle(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if 'conv2' in self.mixstyle_layers:
            x = self.mixstyle(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if 'conv3' in self.mixstyle_layers:
            x = self.mixstyle(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class StyleAugmentor(nn.Module):
    def __init__(self, beta_mode=False, batch_ratio=0.5):
        super(StyleAugmentor,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()

        self.beta_mode = beta_mode
        self.batch_ratio = batch_ratio
        # load checkpoints:
        checkpoint_ghiasi = torch.load(join(dirname(__file__),'styleaugment/styleaug/checkpoints/checkpoint_transformer.pth'))
        checkpoint_stylepredictor = torch.load(join(dirname(__file__),'styleaugment/styleaug/checkpoints/checkpoint_stylepredictor.pth'))
        checkpoint_embeddings = torch.load(join(dirname(__file__),'styleaugment/styleaug/checkpoints/checkpoint_embeddings.pth'))
        
        # load mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)
        self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean']
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        
        # compute SVD of covariance matrix:
        u, s, vh = np.linalg.svd(self.cov.numpy())
        
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float()
    
    def sample_embedding(self,n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100) # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self,x,alpha=0.5,downsamples=0,embedding=None,useStylePredictor=True):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.

        n_mean = torch.tensor([0.5, 0.5, 0.5])
        n_mean = n_mean.to(device = torch.device('cuda:0'))
        n_std = torch.tensor([0.5, 0.5, 0.5])
        n_std = n_std.to(device = torch.device('cuda:0'))

        if not self.training:
            x = F.interpolate(x, size=[32, 32])
            x.sub_(n_mean[None, :, None, None]).div_(n_std[None, :, None, None])
            return x

        if self.batch_ratio < 0:
            x = F.interpolate(x, size=[32, 32])
            x.sub_(n_mean[None, :, None, None]).div_(n_std[None, :, None, None])
            return x

        index_select = torch.randperm(x.size(0))[:(int(x.size(0) * self.batch_ratio))]
        index_select = index_select.to(device = torch.device('cuda:0'))
        x_select = torch.index_select(x, 0, index_select)

        # split x_select into 32s 
        x_select_splits = torch.split(x_select, 32, dim=0)

        # Beta Distribution
        if self.beta_mode:
            beta = torch.distributions.Beta(0.1, 0.1)
            lmda = beta.sample()
        else:
            lmda = alpha

        restyled_lists = []

        for split in x_select_splits:

            base = self.stylePredictor(split) if useStylePredictor else self.imagenet_embedding

            if downsamples:
                assert(split.size(2) % 2**downsamples == 0)
                assert(split.size(3) % 2**downsamples == 0)
                for i in range(downsamples):
                    x_select = nn.functional.avg_pool2d(split,2)

            #if embedding is None:
            embedding = self.sample_embedding(split.size(0))

            # interpolate style embeddings:
            embedding = embedding.to(device = torch.device('cuda:0'))
            embedding = lmda*embedding + (1-lmda)*base

            with torch.no_grad():
                split_restyled = self.ghiasi(split,embedding)

            if downsamples:
                split_restyled = nn.functional.upsample(split_restyled,scale_factor=2**downsamples,mode='bilinear')

            restyled_lists.append(split_restyled)
            del split_restyled
            torch.cuda.empty_cache()

        restyled = torch.cat(restyled_lists, dim=0)
        x[index_select,:] = restyled

        x = F.interpolate(x, size=[32, 32])

        x.sub_(n_mean[None, :, None, None]).div_(n_std[None, :, None, None])
        
        return x.detach()


@BACKBONE_REGISTRY.register()
def cnn_digitsdg_syn(pretrained=True, **kwargs):

    model1 = StyleAugmentor()
    model2 = ConvNet(c_hidden=64)

    init_network_weights(model2, init_type='kaiming')
    model = nn.Sequential(model1,model2)
    model.out_features = 256
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_mnist(pretrained=True, **kwargs):

    model1 = StyleAugmentor()
    model2 = ConvNet(c_hidden=64)

    init_network_weights(model2, init_type='kaiming')
    model = nn.Sequential(model1,model2)
    model.out_features = 256
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_mnist_m(pretrained=True, **kwargs):

    model1 = StyleAugmentor(beta_mode=True)
    model2 = ConvNet(c_hidden=64)

    init_network_weights(model2, init_type='kaiming')
    model = nn.Sequential(model1,model2)
    model.out_features = 256
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_svhn(pretrained=True, **kwargs):

    model1 = StyleAugmentor(beta_mode=True)
    model2 = ConvNet(c_hidden=64)

    init_network_weights(model2, init_type='kaiming')
    model = nn.Sequential(model1,model2)
    model.out_features = 256
    return model