# 
# Custom layers
#

# Requirements
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

#
# Layers
#

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ResNetLayer(torch.nn.Module):
    def __init__(self, dim, depth, kernel_size):
        super(ResNetLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            *[torch.nn.Sequential(
                    Residual(torch.nn.Sequential(
                                torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
                                torch.nn.BatchNorm2d(dim),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
                                torch.nn.BatchNorm2d(dim),
                    ),
                    ),
                    torch.nn.ReLU(inplace=True),
            ) for _ in range(depth)]
        )


    def forward(self, batch):
        return self.layer(batch)


class ConvMixerLayer(torch.nn.Module):
    def __init__(self, dim, depth, kernel_size=9):
        super().__init__()
        self.layer = torch.nn.Sequential(
            *[torch.nn.Sequential(
                    Residual(torch.nn.Sequential(
                        torch.nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        torch.nn.GELU(),
                        torch.nn.BatchNorm2d(dim)
                    )),
                    torch.nn.Conv2d(dim, dim, kernel_size=1),
                    torch.nn.GELU(),
                    torch.nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )

    def forward(self, batch):
        return self.layer(batch)

    
    
#
# TorchVision toolkit (https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/components.py)
#

class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))
    
class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecoderBottleneck(nn.Module):
    """ResNet bottleneck, but convs replaced with resize convs."""

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(self, block, layers, latent_dim, input_height, n_channels):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 1024 * block.expansion
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 7 * 7)

        self.layer1 = self._make_layer(block, 512, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 256, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 128, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
        self.layer5 = self._make_layer(block, 32, layers[4], scale=2)
        self.layer6 = self._make_layer(block, 32, layers[5])

        self.upscale = Interpolate(size=self.input_height)

        self.conv1 = nn.Conv2d(32 * block.expansion, n_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        #print("Shape after linear projection:",x.shape)
        x = x.view(x.size(0), -1, 7, 7)
        #print("Shape after reshape:",x.shape)
        x = self.layer1(x)
        #print("Shape after layer 1:",x.shape)
        x = self.layer2(x)
        #print("Shape after layer 2:",x.shape)
        x = self.layer3(x)
        #print("Shape after layer 3:",x.shape)
        x = self.layer4(x)
        #print("Shape after layer 4:",x.shape)
        x = self.layer5(x)
        #print("Shape after layer 5:",x.shape)
        x = self.layer6(x)
        #print("Shape after layer 6:",x.shape)
        x = self.upscale(x)
        #print("Shape after interpolation:",x.shape)
        x = self.conv1(x)
        return x
    

#
# Encoder
#

class Encoder(torch.nn.Module):
    def __init__(self,
                model_backbone:str='facebook/deit-small-patch16-224',
                style:str='ConvMixer',
                n_channels:int=3,
                latent_dim:int=256,
                dim:int=768,
                depth:int=16,
                kernel_size:int=9,
                patch_size:int=7,
                ):
        super().__init__()
        self.style = style
        if self.style=='ConvMixer':
            self.backbone = torch.nn.Sequential(
                                torch.nn.Conv2d(n_channels, dim, kernel_size=patch_size, stride=patch_size),
                                torch.nn.GELU(),
                                torch.nn.BatchNorm2d(dim),
                                ConvMixerLayer(dim, depth, kernel_size),
                                torch.nn.AdaptiveAvgPool2d((1,1)),
                                torch.nn.Flatten(),
                            )
            self.embedding = torch.nn.Linear(dim, latent_dim)
            self.log_concentration = torch.nn.Linear(dim, 1)
        elif self.style=='ResNet':
            self.backbone = torch.nn.Sequential(
                                torch.nn.Conv2d(n_channels, dim, kernel_size=patch_size, stride=patch_size),
                                torch.nn.GELU(),
                                torch.nn.BatchNorm2d(dim),
                                ResNetLayer(dim, depth, kernel_size=kernel_size),
                                torch.nn.AdaptiveAvgPool2d((1,1)),
                                torch.nn.Flatten(),
                            )
            self.embedding = torch.nn.Linear(dim, latent_dim)
            self.log_concentration = torch.nn.Linear(dim, 1)
        else:
            config = AutoConfig.from_pretrained(model_backbone)
            self.hidden_size = config.hidden_size
            self.cnn = torch.nn.Conv2d(n_channels, config.num_channels, 7, padding="same") if n_channels!=config.num_channels else torch.nn.Identity()
            self.extractor = AutoModel.from_config(config)
            self.embedding = torch.nn.Linear(self.hidden_size, latent_dim)
            self.log_concentration = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, batch):
        # Use backbone
        if self.style!='HuggingFace':
            out = self.backbone(batch)
        else:
            out = self.cnn(batch)
            if hasattr(self.extractor, 'pooler_output'):
                out = self.extractor(out).pooler_output
            else:
                out = torch.mean(self.extractor(out).last_hidden_state, dim=1)
        # Get output
        output = {
            "embedding":self.embedding(out),
            "log_covariance":self.log_concentration(out),
        }
        return output


#
# Decoder
#

class Decoder(torch.nn.Module):
    def __init__(self,
                style:str='ConvMixer',
                img_size:int=224,
                n_channels:int=3,
                latent_dim:int=256,
                dim:int=768,
                depth:int=4,
                kernel_size:int=9,
                patch_size:int=7,
                ):
        super().__init__()
        # Parameters
        self.dim = dim
        self.patch_size = patch_size
        size_list = [img_size//2**i if ((img_size%2**i==0) & ((img_size/2**i)>=self.patch_size)) else np.inf for i in range(10)][::-1]
        size_list = [elem for elem in size_list if elem<np.inf]
        assert min(size_list)==self.patch_size, f"Patch size {self.patch_size} must be a power of two of final image size {img_size}."
        filter_list = [self.dim//(2**i) for i in range(len(size_list))]
        # Layers
        self.expansion = torch.nn.Linear(latent_dim, self.dim*self.patch_size**2)
        if style=='ConvMixer':
            self.backbone = torch.nn.Sequential(
                                *[torch.nn.Sequential(
                                    ConvMixerLayer(filter_list[i], depth, kernel_size), # Process
                                    torch.nn.ConvTranspose2d(filter_list[i], filter_list[i+1], kernel_size=2, stride=2), # Upsample
                                    torch.nn.GELU(),
                                    torch.nn.BatchNorm2d(filter_list[i+1])
                                ) for i in range(len(size_list)-1)
                                ],
                                torch.nn.Conv2d(filter_list[-1], n_channels, kernel_size = 5, padding='same')
                            )
        elif style=='ResNet':
            self.backbone = torch.nn.Sequential(
                                *[torch.nn.Sequential(
                                    ResNetLayer(filter_list[i], depth), # Process
                                    torch.nn.ConvTranspose2d(filter_list[i], filter_list[i+1], kernel_size=2, stride=2), # Upsample
                                    torch.nn.GELU(),
                                    torch.nn.BatchNorm2d(filter_list[i+1])
                                ) for i in range(len(size_list)-1)
                                ],
                                torch.nn.Conv2d(filter_list[-1], n_channels, kernel_size = 5, padding='same'),
                            )
        elif style=='ResNet18':
            self.backbone = ResNetDecoder(DecoderBlock, [depth]*6, latent_dim, img_size, n_channels)
        elif style=='ResNet50':
            self.backbone = ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3, 3, 3], latent_dim, img_size, n_channels)
        self.style = style

    def forward(self, batch):
        if self.style in ['Convmixer', 'ResNet']:
            out = self.expansion(batch)
            out = torch.reshape(out, (-1, self.dim, self.patch_size, self.patch_size))
            output = self.backbone(out)
            return {'reconstruction':output}
        elif self.style in ['ResNet18', 'ResNet50']:
            output = self.backbone(batch)
            return {'reconstruction':output}
        else:
            return None
