from .build import build_backbone, BACKBONE_REGISTRY # isort:skip
from .backbone import Backbone # isort:skip

from .resnet import (
    resnet18_pacs_photo, resnet18_pacs_art_painting, resnet18_pacs_cartoon, resnet18_pacs_sketch, 
    resnet18_office_art, resnet18_office_product, resnet18_office_real_world, resnet18_office_clipart
)
from .cnn_digitsdg import (
    cnn_digitsdg_syn, cnn_digitsdg_mnist, cnn_digitsdg_mnist_m, cnn_digitsdg_svhn
)