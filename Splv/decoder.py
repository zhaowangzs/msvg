from .utils import fill_up_weights
from external_pkgs.deformable import ModulatedDeformConvWithOff

from torch import nn

BN_MOMENTUM = 0.1


class Deconder(nn.Module):
    def __init__(self, inplanes_list, planes_list=(256, 128, 64), mode='FCN', skip_dim=256):
        super(Deconder, self).__init__()

        self.inplanes_list = inplanes_list
        self.planes = planes_list
        self.mode = mode
        self.skip_dim =skip_dim
        self.up_layer = []
        if self.mode == 'FCN':
            self.inplanes = self.inplanes_list[-1]
            for i in range(len(self.inplanes_list) - 1):
                self.up_layer.append(self._make_upsample(self.inplanes, self.planes[i], 4))
                self.inplanes = self.planes[i]
        elif self.mode == 'FPN':
            self.later_layer = []
            for i in range(len(self.inplanes_list) - 1):
                self.later_layer.append(nn.Conv2d(self.inplanes_list[-1 - i], self.skip_dim, 1))
                self.up_layer.append(self._make_upsample(self.skip_dim, self.skip_dim, 4))
            self.later_layer = nn.ModuleList(self.later_layer)
        else:
            raise NotImplementedError

        self.up_layer = nn.ModuleList(self.up_layer)

    def _make_upsample(self, inplanes, planes, num_kernels, deconv_with_bias=False):
        layers = []
        fc = ModulatedDeformConvWithOff(
            inplanes, planes,
            kernel_size=3, deformable_groups=1,
        )
        kernel, padding, output_padding = self.get_deconv_cfg(num_kernels)
        up = nn.ConvTranspose2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=deconv_with_bias)
        fill_up_weights(up)
        layers.append(fc)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def init_weights(self):
        for deconv_layers in self.up_layer:
            for name, m in deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        num_up_layers = len(self.up_layer)
        if self.mode == 'FPN':
            out = self.later_layer[0](x[-1])
            for i in range(num_up_layers):
                out = self.up_layer[i](out)
                out = out + self.later_layer[i + 1](x[-2 - i])
        elif self.mode == 'FCN':
            out = x[-1]
            for i in range(num_up_layers):
                out = self.up_layer[i](out)
        return out
