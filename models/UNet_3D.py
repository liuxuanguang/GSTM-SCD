import torch
import torch.nn as nn


def conv_block(in_dim, middle_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def center_in(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True)
    )


def center_out(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
    )


def up_conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, timesteps, dropout):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.timesteps = timesteps

        feats = 16
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        # 添加ceil_mode=True确保输出尺寸向上取整
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)

        # 修改输出层
        self.final = nn.Sequential(
            nn.Conv3d(feats * 2, feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feats),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(feats, n_classes, kernel_size=1)
        )

        self.dropout = nn.Dropout3d(p=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # 编码路径
        en3 = self.en3(x)
        pool_3 = self.pool_3(en3)

        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)

        # 中心处理
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)

        # 动态尺寸调整
        if center_out.shape[2:] != en4.shape[2:]:
            center_out = torch.nn.functional.interpolate(
                center_out, size=en4.shape[2:],
                mode='trilinear', align_corners=True
            )

        # 拼接并处理
        concat4 = torch.cat([center_out, en4], dim=1)
        dc4 = self.dc4(concat4)

        # 上采样
        trans3 = self.trans3(dc4)

        # 动态尺寸调整
        if trans3.shape[2:] != en3.shape[2:]:
            trans3 = torch.nn.functional.interpolate(
                trans3, size=en3.shape[2:],
                mode='trilinear', align_corners=True
            )

        # 拼接并处理
        concat3 = torch.cat([trans3, en3], dim=1)
        dc3 = self.dc3(concat3)

        # 应用Dropout
        dc3 = self.dropout(dc3)

        # 最终输出
        final = self.final(dc3)
        final = self.logsoftmax(final)

        return final


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channel=4, n_classes=8, timesteps=6, dropout=0.2).to(device)
    input_tensor = torch.randn(1, 4, 6, 512, 512).to(device)
    output = model(input_tensor)
    from thop import profile
    FLOPs, Params = profile(model, (input_tensor,))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
