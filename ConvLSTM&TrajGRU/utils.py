import torch.nn as nn
from collections import OrderedDict

def make_layers(blocks):
    '''
    :param blocks: a dic
    :return: Models
    '''
    layers = []
    for layer_name, v in blocks.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0],
                                 stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif "deconv" in layer_name:
            layer = nn.ConvTranspose2d(in_channels=v[0],
                              out_channels=v[1],
                              kernel_size=v[2],
                              stride=v[3],
                              padding=v[4])
            layers.append((layer_name, layer))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif "conv" in layer_name:
            layer = nn.Conv2d(in_channels=v[0],
                              out_channels=v[1],
                              kernel_size=v[2],
                              stride=v[3],
                              padding=v[4])
            layers.append((layer_name, layer))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))