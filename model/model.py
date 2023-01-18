import torch
import torch.nn as nn
from loss import LossFunction
import kornia
from model.xiaoye import Wnet
import torch.nn.functional as F


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # input_max = torch.max(input, dim=1, keepdim=True)[0]
        # input_img = torch.cat((input_max, input), dim=1)
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + input #torch.max(input, dim=1, keepdim=True)[0]
        illu = torch.clamp(illu, 0.0001, 1)

        # R = torch.clamp(R, 0.0001, 1)
        # L = torch.clamp(L, 0.0001, 1)
        return illu


class EnhanceNetwork1(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork1, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # input_max = torch.max(input, dim=1, keepdim=True)[0]
        # input_img = torch.cat((input_max, input), dim=1)
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = conv(fea)
        fea = self.out_conv(fea)

        # R = torch.clamp(R, 0.0001, 1)
        # L = torch.clamp(L, 0.0001, 1)
        return fea
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out
def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)
def smooth(input_I, input_R):
    input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
    input_R = torch.unsqueeze(input_R, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.net3 = EnhanceNetwork(5, 8)#EnhanceNetwork(5, 8)
        self._criterion = LossFunction()
        self.gamma_net = Wnet(3)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1loss = torch.nn.L1Loss()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input, label):
        beta, gamma = self.gamma_net(input)
        # print(beta, gamma)
        i = self.net3(input)
        i_h = self.net3(label)
        i_enhance = beta * torch.pow(i, gamma)
        enhance_image = i_enhance * (input / i)
        enhance_image = torch.clamp(enhance_image, 0, 1)

        return enhance_image, i, i_enhance, input / i, i_h

    def _loss(self, input, high, epoch):
        enhance_image, i, i_e, r, i_h = self(input, high)


        loss = self._criterion(high, enhance_image)
        # loss += 0.001 * self._criterion(i_e, torch.max(high, dim=1, keepdim=True)[0])
        # loss += 0.001 * self._criterion(i, torch.max(input, dim=1, keepdim=True)[0])

        return loss



class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = Network()#EnhanceNetwork(layers=3, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def forward(self, input):
        enhance_image, i, i_enhance, r = self.enhance(input, input)
        return enhance_image, i, i_enhance, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

