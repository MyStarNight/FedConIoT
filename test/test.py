import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from src.websocket_client import MyWebsocketClientWorker
from src.nn_model import ConvNet1D, loss_fn
from src.my_utils import generate_kwarg, generate_command_dict
from src.config import Config
import torch
from datetime import datetime
import asyncio
import logging
from src.model_evaluation import evaluate
import pickle
from torch.utils.data import TensorDataset, DataLoader
from src.model_evaluation import evaluate
from src.nn_model import model_to_device, aggregate_models
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_1 = ConvNet1D(input_size=400, num_classes=7)
model_2 = ConvNet1D(input_size=400, num_classes=7)

traced_model1 = torch.jit.trace(model_1, torch.zeros([1, 400, 3], dtype=torch.float))
traced_model2 = torch.jit.trace(model_2, torch.zeros([1, 400, 3], dtype=torch.float))

data_path = '../../Dataset/HAR/HAR_datasets.pkl'
with open(data_path, 'rb') as f:
    HAR_datasets = pickle.load(f)


selected_data = []
selected_target = []
for user in [1, 2, 3, 4]:
    selected_data.append(HAR_datasets[user][:][0])
    selected_target.append(HAR_datasets[user][:][1])

selected_data_tensor = torch.cat(selected_data, dim=0)
selected_target_tensor = torch.cat(selected_target, dim=0)
train_data = TensorDataset(selected_data_tensor, selected_target_tensor)

train_model1 = model_to_device(traced_model1, device)
train_model2 = model_to_device(traced_model2, device)

optimizer1 = optim.Adam(train_model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(train_model2.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=32)

for cur_round in range(1,10+1):

    for _ in range(3):
        if cur_round % 2 == 1:
            for data, target in train_loader:
                optimizer1.zero_grad()
                output = train_model1(data.to(device))
                loss = loss_fn(output, target.to(device))
                loss.backward()
                optimizer1.step()
        else:
            for data, target in train_loader:
                optimizer2.zero_grad()
                output = train_model2(data.to(device))
                loss = loss_fn(output, target.to(device))
                loss.backward()
                optimizer2.step()

    if cur_round % 2 == 0:
        print("local")
        evaluate(train_model1)
        evaluate(train_model2)

        traced_model1 = torch.jit.trace(
            model_to_device(traced_model1, 'cpu'),
            torch.zeros([1, 400, 3], dtype=torch.float)
        )
        traced_model2 = torch.jit.trace(
            model_to_device(traced_model2, 'cpu'),
            torch.zeros([1, 400, 3], dtype=torch.float)
        )

        federated_model = aggregate_models({1:traced_model1, 2:train_model2}, {1:1, 2:1})
        print("federated")
        evaluate(federated_model)
        train_model1.load_state_dict(federated_model.state_dict())
        train_model2.load_state_dict(federated_model.state_dict())
        print()