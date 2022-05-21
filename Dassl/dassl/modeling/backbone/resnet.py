import torch.nn as nn
import torch.utils.model_zoo as model_zoo

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

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle(x)
        return self.layer4(x)

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)
        #return f

def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


class StyleAugmentor(nn.Module):
    def __init__(self, beta_mode=False, batch_ratio=0.5, alpha=0.5):
        super(StyleAugmentor,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()
        self.beta_mode = beta_mode
        self.batch_ratio = batch_ratio
        self.alpha = alpha
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

    def forward(self,x,downsamples=0,embedding=None,useStylePredictor=True):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.

        n_mean = torch.tensor([0.485, 0.456, 0.406])
        n_mean = n_mean.to(device = torch.device('cuda:0'))
        n_std = torch.tensor([0.229, 0.224, 0.225])
        n_std = n_std.to(device = torch.device('cuda:0'))

        if not self.training:
            x.sub_(n_mean[None, :, None, None]).div_(n_std[None, :, None, None])
            return x

        if self.batch_ratio < 0:
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
            lmda = self.alpha

        restyled_lists = []

        for split in x_select_splits:

            base = self.stylePredictor(split) if useStylePredictor else self.imagenet_embedding

            if downsamples:
                assert(split.size(2) % 2**downsamples == 0)
                assert(split.size(3) % 2**downsamples == 0)
                for i in range(downsamples):
                    split = nn.functional.avg_pool2d(split,2)

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

        x.sub_(n_mean[None, :, None, None]).div_(n_std[None, :, None, None])
        
        return x.detach()


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
########################PACS###############################
@BACKBONE_REGISTRY.register()
def resnet18_pacs_photo(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model1 = StyleAugmentor()

    model2 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_pacs_art_painting(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model1 = StyleAugmentor()

    model2 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_pacs_cartoon(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model1 = StyleAugmentor()

    model2 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_pacs_sketch(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model1 = StyleAugmentor(beta_mode=True)

    model2 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model


######################Office Home###################
@BACKBONE_REGISTRY.register()
def resnet18_office_art(pretrained=True, **kwargs):

    model1 = StyleAugmentor(batch_ratio=0.3)

    model2 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_office_product(pretrained=True, **kwargs):

    model1 = StyleAugmentor(batch_ratio=0.2)

    model2 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_office_real_world(pretrained=True, **kwargs):

    model1 = StyleAugmentor(batch_ratio=0.1)

    model2 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model

@BACKBONE_REGISTRY.register()
def resnet18_office_clipart(pretrained=True, **kwargs):

    from dassl.modeling.ops import MixStyle

    model1 = StyleAugmentor(beta_mode=True, batch_ratio=0.5)

    model2 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model2, model_urls['resnet18'])

    model = nn.Sequential(model1,model2)
    model.out_features = 512

    return model