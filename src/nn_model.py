import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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


def aggregate_models(model_dict, sample_dict):
    """
    Aggregate models using custom weights based on the sample counts.
    This function supports both nn.Module and torch.jit.ScriptModule.

    Parameters:
    model_dict (dict): A dictionary containing the models to be aggregated with format {id: model}.
    sample_dict (dict): A dictionary containing the sample counts with format {id: sample_count}.

    Returns:
    aggregated_model: The aggregated model.
    """

    # Determine the total number of samples
    total_samples = sum(sample_dict.values())

    # Initialize the aggregated model state dict with the first model's state dict
    first_model_id = next(iter(model_dict))
    first_model = model_dict[first_model_id]

    if isinstance(first_model, torch.jit.ScriptModule):
        # For ScriptModule, we need to recompile the module structure
        aggregated_model = torch.jit.script(first_model)
        aggregated_model_state_dict = {k: torch.zeros_like(v) for k, v in first_model.state_dict().items()}
    else:
        # For nn.Module, we can use deepcopy
        aggregated_model = copy.deepcopy(first_model)
        aggregated_model_state_dict = {k: torch.zeros_like(v) for k, v in aggregated_model.state_dict().items()}

    # Aggregate the models
    for model_id, model in model_dict.items():
        weight = sample_dict[model_id] / total_samples
        model_state_dict = model.state_dict()

        for key in aggregated_model_state_dict.keys():
            aggregated_model_state_dict[key] += weight * model_state_dict[key]

    # Load the aggregated state dict into the aggregated model
    if isinstance(aggregated_model, torch.jit.ScriptModule):
        aggregated_model = torch.jit.script(aggregated_model)

    aggregated_model.load_state_dict(aggregated_model_state_dict)

    return aggregated_model
