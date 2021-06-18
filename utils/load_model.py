import torch
import torch.nn as nn


def count_model_params(model, is_trainable=True):
    if is_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


class SkipConnBlock(nn.Module):
    def __init__(self, cin, cout):
        super(SkipConnBlock, self).__init__()
        self.cin = cin
        self.cout = cout

        self.conv1 = self.conv_block(
            c_in=cin, c_out=cout, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv2 = self.conv_block(
            c_in=cout,
            c_out=cout,
            kernel_size=3,
            stride=1,
            groups=cout,
            padding=1,
            bias=False,
        )

        self.conv3 = self.conv_block(
            c_in=cin, c_out=cout, kernel_size=1, stride=1, bias=False
        )

        # dropout and relu layers.
        self.drop = nn.Dropout2d(0.125)
        self.relu = nn.ReLU()

    def forward(self, x):
        # store initial input for later use
        og = x

        # 2 convolution operations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.cin == self.cout:
            x += og
        else:
            og = self.conv3(og)
            x += og

        x = self.drop(x)
        x = self.relu(x)

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
        )

        return seq_block


class ModelCL(nn.Module):
    def __init__(self, num_classes):
        super(ModelCL, self).__init__()

        self.num_classes = num_classes

        self.sizes_block_1 = [3, 16, 16, 16, 16]
        self.sizes_block_2 = [16, 32, 32, 32, 32]
        self.sizes_block_3 = [32, 64, 64, 64, 64]
        self.sizes_block_4 = [64]

        ####### block 1 #######
        self.block_1 = nn.Sequential(
            *[
                self.conv_block(i, o, kernel_size=3, stride=1, padding=0, dilation=1)
                for i, o in zip(self.sizes_block_1, self.sizes_block_1[1:])
            ]
        )  # 24

        ####### block 2 #######
        self.block_2 = nn.Sequential(
            *[
                self.conv_block(i, o, kernel_size=3, stride=1, padding=0, dilation=1)
                for i, o in zip(self.sizes_block_2, self.sizes_block_2[1:])
            ]
        )  # 18

        ####### block 3 #######
        self.block_3 = nn.Sequential(
            *[
                self.conv_block(i, o, kernel_size=3, stride=1, padding=2, dilation=2)
                for i, o in zip(self.sizes_block_3, self.sizes_block_3[1:])
            ]
        )  # 18

        ####### block 4 #######
        self.block_4 = nn.Sequential(
            *[
                self.conv_block(i, o, kernel_size=3, stride=1, padding=0, dilation=2)
                for i, o in zip(self.sizes_block_4, self.sizes_block_4[1:])
            ]
        )  # 10

        self.dsc = self.depth_conv(
            c_in=self.sizes_block_4[-1],
            c_out=self.sizes_block_4[-1],
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.final_conv = nn.Conv2d(
            in_channels=self.sizes_block_4[-1],
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.gap = nn.AvgPool2d(kernel_size=10)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = self.dsc(x)

        x = self.gap(x)

        x = self.final_conv(x)
        x = x.squeeze()

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.0125),
        )

        return seq_block

    def depth_conv(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in, out_channels=c_in, groups=c_in, bias=False, **kwargs
            ),
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.0125),
        )

        return seq_block


if __name__ == "__main__":

    from torchinfo import summary

    # classification models
    print("Classification Model (num params):")
    model = ModelCL(num_classes=10)
    print(summary(model, input_size=(2, 3, 32, 32)))
    # inp = torch.rand(2, 3, 32, 32)
    # print(f"Input: {inp.shape}")
    # print(f"Output: {model(inp).shape}")
    # print(f"Num parameters: {count_model_params(model)}")
