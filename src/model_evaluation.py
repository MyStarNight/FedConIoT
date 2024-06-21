import torch
import pickle
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse


def evaluate(model, device='cpu'):
    data_path = '../Dataset/HAR/HAR_datasets.pkl'
    with open(data_path, 'rb') as f:
        HAR_datasets = pickle.load(f)

    selected_data = []
    selected_target = []
    for user in [1, 2, 3, 4]:
        selected_data.append(HAR_datasets[user][:][0])
        selected_target.append(HAR_datasets[user][:][1])

    selected_data_tensor = torch.cat(selected_data, dim=0)
    selected_target_tensor = torch.cat(selected_target, dim=0).argmax(dim=1)
    test_data = TensorDataset(selected_data_tensor, selected_target_tensor)

    model.eval()

    test_loader = DataLoader(test_data, batch_size=32)

    test_correct = 0
    test_total = 0
    class_labels = [0, 1, 2, 3, 4, 5, 6]  # Replace with your actual class labels

    # Initialize a dictionary to store correct and total predictions for each class
    class_correct = {label: 0 for label in class_labels}
    class_total = {label: 0 for label in class_labels}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

            # Update class-wise correct and total predictions
            for label in class_labels:
                class_mask = targets == label
                class_correct[label] += (predicted[class_mask] == targets[class_mask]).sum().item()
                class_total[label] += class_mask.sum().item()

    # Print overall accuracy
    print(f'Test Accuracy: {(test_correct / test_total) * 100:.2f}%')

    # Print class-wise accuracy
    # for label in class_labels:
    #     accuracy = (class_correct[label] / class_total[label]) * 100 if class_total[label] != 0 else 0
    #     print(f'Class {label} Accuracy: {accuracy:.2f}%')

