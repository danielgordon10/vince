import torchvision.models
from efficientnet_pytorch import EfficientNet
from torch import nn

from models.building_blocks import resnet as my_resnet

__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet18SiamFCDilated",
    "ResNet50SiamFCDilated",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
]


class Backbone(nn.Module):
    def __init__(self, args, model, final_layer=None):
        super(Backbone, self).__init__()
        self.args = args
        self.model = model
        self.model_children = list(self.model.children())
        self.final_layer = final_layer
        if self.final_layer is None:
            self.final_layer = len(self.model_children)
        if self.final_layer != 0:
            if self.final_layer < 0:
                self.final_layer = len(self.model_children) + final_layer
            print("Output layer in backbone", self.model_children[self.final_layer - 1])
            if self.final_layer < len(self.model_children):
                print("Layer after output layer in backbone", self.model_children[self.final_layer])

        self.output_channels = self.model.output_channels

    def forward(self, x, final_layer=None):
        # final_layer == 0 returns the input
        # final_layer == 1 returns the output after the first layer
        # final_layer == -1 returns the output before the last layer
        if final_layer is None:
            if self.final_layer is not None:
                final_layer = self.final_layer
            else:
                final_layer = len(self.model_children) + 1
        if final_layer < 0:
            final_layer = len(self.model_children) + final_layer
        for cc, child in enumerate(self.model.children()):
            if cc >= final_layer:
                return x
            x = child(x)
        return x


class ResNet18(Backbone):
    def __init__(self, args, final_layer=None):
        model = torchvision.models.resnet18(args.use_imagenet_weights)
        model.output_channels = 512
        super(ResNet18, self).__init__(args, model, final_layer)


class ResNet34(Backbone):
    def __init__(self, args, final_layer=None):
        model = torchvision.models.resnet34(args.use_imagenet_weights)
        model.output_channels = 512
        super(ResNet34, self).__init__(args, model, final_layer)


class ResNet50(Backbone):
    def __init__(self, args, final_layer=None):
        model = torchvision.models.resnet50(args.use_imagenet_weights)
        model.output_channels = 2048
        super(ResNet50, self).__init__(args, model, final_layer)


class ResNet50SiamFCDilated(Backbone):
    def __init__(self, args, final_layer=None):
        model = torchvision.models.resnet50(args.use_imagenet_weights, replace_stride_with_dilation=[False, True, True])
        model.output_channels = 2048
        super(ResNet50SiamFCDilated, self).__init__(args, model, final_layer)


class ResNet18SiamFCDilated(Backbone):
    def __init__(self, args, final_layer=None):
        model = my_resnet.resnet18(args.use_imagenet_weights, replace_stride_with_dilation=[False, True, True])
        model.output_channels = 512
        super(ResNet18SiamFCDilated, self).__init__(args, model, final_layer)


class EfficientNetB0(Backbone):
    def __init__(self, args, final_layer=None):
        model = EfficientNet.from_name("efficientnet-b0")
        model.output_channels = 1280
        super(EfficientNetB0, self).__init__(args, model, final_layer)


class EfficientNetB1(Backbone):
    def __init__(self, args, final_layer=None):
        model = EfficientNet.from_name("efficientnet-b1")
        model.output_channels = 1280
        super(EfficientNetB1, self).__init__(args, model, final_layer)


class EfficientNetB2(Backbone):
    def __init__(self, args, final_layer=None):
        model = EfficientNet.from_name("efficientnet-b2")
        model.output_channels = 1408
        super(EfficientNetB2, self).__init__(args, model, final_layer)


class EfficientNetB3(Backbone):
    def __init__(self, args, final_layer=None):
        model = EfficientNet.from_name("efficientnet-b3")
        model.output_channels = 1536
        super(EfficientNetB3, self).__init__(args, model, final_layer)


class EfficientNetB4(Backbone):
    def __init__(self, args, final_layer=None):
        model = EfficientNet.from_name("efficientnet-b4")
        model.output_channels = 1792
        super(EfficientNetB4, self).__init__(args, model, final_layer)
