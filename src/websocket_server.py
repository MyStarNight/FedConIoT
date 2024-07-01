from syft.workers.websocket_server import WebsocketServerWorker
import time
import torch
from src.websocket_client import MyWebsocketClientWorker
import syft as sy
from src.nn_model import ConvNet1D, loss_fn, model_to_device, aggregate_models, loss_fn_test
from src import my_utils
from typing import Dict, List
from src.pull_and_push import MyTrainConfig
from syft.workers.abstract import AbstractWorker
from syft.generic.pointers.object_wrapper import ObjectWrapper
from datetime import datetime
from src.config import Config
from src.pull_and_push import PullAndPush
from syft.frameworks.torch.fl import utils
import numpy as np
from src.model_evaluation import evaluate


class MyWebsocketServerWorker(WebsocketServerWorker):

    def __init__(self, hook, host: str, port: int, id, verbose):
        super().__init__(hook=hook, host=host, port=port, id=id, verbose=verbose)
        self.owner = sy.hook.local_worker
        self.child_nodes: List[MyWebsocketClientWorker] = []

        # 训练部分的参数
        self.traced_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_config = Config(epochs=3, optimizer_args={'lr': 0.001})
        self.p_p = PullAndPush()
        self.train_state = False

        self.model_dict = {}
        self.model_id = my_utils.device_id_to_model_id(self.id)

        self.sample_length = 0
        self.sample_dict = {}

    def add_dataset(self, dataset, key: str):
        super().add_dataset(dataset, key)
        self.sample_length = len(list(self.datasets.values())[0])
        self.sample_dict = {self.id: self.sample_length}

    def total_sample_length(self):
        return np.sum(list(self.sample_dict.values()))

    def connect_child_nodes(self, forward_device_id: list, port=9292):
        for ID in forward_device_id:
            ip = my_utils.id_mapping_client_device[ID]
            kwargs_websocket = {"hook": self.hook, "host": ip, "port": port, "id": ID}
            self.child_nodes.append(MyWebsocketClientWorker(**kwargs_websocket))

    def clear_child_nodes(self):
        for child_node in self.child_nodes:
            child_node.clear_objects_remote()

    def close_child_nodes(self):
        for child_node in self.child_nodes:
            child_node.close()
        self.child_nodes = []

    def command(self, command: dict):
        message = self.create_worker_command_message(**command)
        serialized_message = sy.serde.serialize(message)

        response_list = []
        for child_node in self.child_nodes:
            response = self._send_msg(serialized_message, child_node)
            response_list.append(response)

    @staticmethod
    def test():
        print("test start")
        time.sleep(5)
        print("test end")

    def model_initialization(self):
        model = ConvNet1D(input_size=400, num_classes=7)
        self.traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float))
        self.train_state = True

    def model_dissemination(self, forward_device_id: list, port=9292):
        """
        模型下发
        :param forward_device_id: 需要转发的字节点信息
        :param port: 端口号，默认为9292
        :return:
        """
        # 连接所有子节点
        self.connect_child_nodes(forward_device_id, port)

        # 发送模型给所有的子节点
        for child_node in self.child_nodes:
            if self.traced_model is not None:
                obj_ptr, obj_id = self.p_p.send(self.traced_model, child_node)
                child_node.command({
                    "command_name": "model_configuration",
                    "model_id": obj_id
                })
            else:
                raise ValueError("Traced Model is None")

        # 关闭所有节点的连接
        self.close_child_nodes()

    def model_configuration(self, model_id):
        self.traced_model = self.get_obj(model_id).obj
        self.de_register_obj(self.get_obj(model_id))
        self.train_state = True

    def change_state(self, state=True):
        self.train_state = state

    def check_train_state(self):
        if self.train_state is not True:
            raise ValueError("Train State is False")

    def train(self, dataset_key: str, test=False):

        # 检测是否可以进行训练
        self.check_train_state()
        print(f'\n{self.device} is available.')
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        model = model_to_device(self.traced_model, self.device)

        # 为训练做准备
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.train_config.lr
        )

        model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None

        print(f"{datetime.now()}: Training Start")
        # print(next(self.model.parameters()).device)

        for _ in range(self.train_config.epochs):
            for (data, target) in data_loader:
                # Set gradients to zero
                optimizer.zero_grad()

                # Update model
                output = model(data.to(self.device))
                loss = loss_fn(target=target.to(self.device), pred=output)
                loss.backward()
                optimizer.step()

        model.eval()
        self.traced_model = torch.jit.trace(
            model_to_device(model, 'cpu'),
            torch.zeros([1, 400, 3], dtype=torch.float)
        )

        if test:
            evaluate(model, self.device)
        print(loss.item())
        print(f"{datetime.now()}: Training End")

        self.train_state = False
        self.model_dict = {self.id: self.traced_model}
        self.sample_dict = {self.id: self.sample_length}

        torch.save(model.state_dict(), 'model.pth')

    def model_collection(self, forward_device_id: list, port=9292, aggregation=False):
        # 连接所有子节点
        self.connect_child_nodes(forward_device_id, port)

        # 收回所有模型
        for child_node in self.child_nodes:
            obj_ptr, obj_id = self.p_p.send(self.traced_model, child_node, self.model_id)
            child_node.command({
                "command_name": "model_storage_and_aggregation",
                "model_id": obj_id,
                "aggregation": aggregation,
                "sample_length": self.total_sample_length()
            })

        # 关闭所有节点的连接
        self.close_child_nodes()

    def model_storage_and_aggregation(self, model_id, aggregation, sample_length):
        device_id = my_utils.model_id_to_device_id(model_id)
        self.model_dict[device_id] = self.get_obj(model_id).obj
        self.sample_dict[device_id] = sample_length
        self.de_register_obj(self.get_obj(model_id))

        if aggregation:
            self.traced_model = aggregate_models(self.model_dict, self.sample_dict)

    def set_federated_model(self, federated_model_id=int(str(1)*11)):
        obj_with_id = ObjectWrapper(id=federated_model_id, obj=self.traced_model)
        self.set_obj(obj_with_id)

    def show_node_state(self):
        print(self.model_dict.keys())
        print(self.sample_dict)

    def model_evaluation(self):
        evaluate(self.traced_model)

    def central_node_storage_clear(self):
        self.model_dict = {}
        self.sample_dict = {}
