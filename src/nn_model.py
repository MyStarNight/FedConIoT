import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((input_size - 3 + 1) // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


@torch.jit.script
def loss_fn(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))


@torch.jit.script
def loss_fn_test(pred, target):
    return F.cross_entropy(pred, target)


def model_to_device(model, device='cpu'):
    """由于在此环境下，模型进行计算以后无法在cpu和cuda之间切换
    因此使用这个函数，进行模型的运算设备切换

    Args:
        model: 模型
        device: cpu or cuda

    Returns:
        traced model to device you set
    """
    new_model = ConvNet1D(input_size=400, num_classes=7).to(device)
    new_model.load_state_dict(model.state_dict())
    return new_model
