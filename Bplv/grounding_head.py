from torch import nn

class GroundingHead(nn.Module):
    def __init__(self, inplanes, v_planes):
        super(GroundingHead, self).__init__()

        self.wh_head = nn.Sequential(nn.Conv2d(inplanes, v_planes,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(v_planes, 2,
                              kernel_size=1, bias=True))

        self.reg_head = nn.Sequential(nn.Conv2d(inplanes, v_planes,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(v_planes, 2,
                              kernel_size=1, bias=True))
        self.ct_head = nn.Sequential(nn.Conv2d(inplanes, v_planes,
                                                kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(v_planes, 1,
                                                kernel_size=1, bias=True))
        self.fill_fc_weights(self.wh_head)
        self.fill_fc_weights(self.reg_head)
        self.ct_head[-1].bias.data.fill_(-2.19)

        self.relu = nn.ReLU()

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        reg = self.reg_head(x)
        wh = self.wh_head(x)
        ct = self.ct_head(x)
        return ct, wh, reg