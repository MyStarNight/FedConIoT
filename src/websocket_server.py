from syft.workers.websocket_server import WebsocketServerWorker
import time
import torch
from src.websocket_client import MyWebsocketClientWorker
import syft as sy
from src.nn_model import ConvNet1D, loss_fn, model_to_device
from src import my_utils
from typing import Dict
from src.pull_and_push import MyTrainConfig
from syft.workers.abstract import AbstractWorker
from syft.generic.pointers.object_wrapper import ObjectWrapper
from datetime import datetime
from src.config import config
from src.pull_and_push import PullAndPush
from syft.frameworks.torch.fl import utils


class MyWebsocketServerWorker(WebsocketServerWorker):

    def __init__(self, hook, host: str, port: int, id, verbose, owner: AbstractWorker = None):
        super().__init__(hook=hook, host=host, port=port, id=id, verbose=verbose)
        self.owner = self.owner = owner if owner else sy.hook.local_worker
        self.child_nodes: Dict[MyWebsocketClientWorker] = []

        # 训练部分的参数
        self.model = None
        self.traced_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.p_p = PullAndPush()
        self.train_state = False

    def connect_child_nodes(self, forward_device_mapping_id: dict, port=9292):
        for ip, ID in forward_device_mapping_id.items():
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
        self.model = ConvNet1D(input_size=400, num_classes=7)
        self.traced_model = torch.jit.trace(self.model, torch.zeros([1, 400, 3], dtype=torch.float))
        self.model.to(self.device)
        self.train_state = True

    def model_dissemination(self, forward_device_mapping_id: dict, port=9292):
        """
        模型下发
        :param forward_device_mapping_id: 需要转发的字节点信息
        :param port: 端口号，默认为9292
        :return:
        """
        # 连接所有子节点
        self.connect_child_nodes(forward_device_mapping_id, port)

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
        self.model = model_to_device(self.traced_model, self.device)
        self.train_state = True

    def check_train_state(self):
        if self.train_state is not True:
            raise ValueError("Train State is False")

    def train(self, dataset_key: str):

        # 检测是否可以进行训练
        self.check_train_state()
        print(f'{self.device} is available.')
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        # 为训练做准备
        self._build_optimizer(
            self.config["optimizer"],
            self.model,
            optimizer_args=self.config['optimizer_args']
        )
        model_id = int(str('1' * 11))

        self.model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.config["shuffle"]
        )

        loss = None
        iteration_count = 0

        print(f"{datetime.now()}: Training Start")
        print(next(self.model.parameters()).device)

        for _ in range(self.train_config.epochs):
            for (data, target) in data_loader:
                # Set gradients to zero
                self.optimizer.zero_grad()

                # Update model
                output = self.model(data.to(self.device))
                loss = loss_fn(target=target.to(self.device), pred=output)
                loss.backward()
                self.optimizer.step()

                # Update and check interation count
                iteration_count += 1
                if iteration_count >= self.train_config.max_nr_batches >= 0:
                    break

        print(f"{datetime.now()}: Training End")

        self.model.eval()
        self.model = model_to_device(self.model, 'cpu')
        self.traced_model = torch.jit.trace(
            self.model,
            torch.zeros([1, 400, 3], dtype=torch.float).to('cpu')
        )

        self.register_obj(ObjectWrapper(id=model_id, obj=self.traced_model))
        self.train_state = False

    def model_collection(self, forward_device_mapping_id: dict, port=9292, model_id=int(str('1' * 11))):
        # 连接所有子节点
        self.connect_child_nodes(forward_device_mapping_id, port)

        # 收回所有模型
        model_dict = {"me": self.traced_model}
        for child_node in self.child_nodes:
            model_dict[child_node.id] = self.owner.request_obj(model_id, child_node)

        # 聚合模型
        self.traced_model = utils.federated_avg(model_dict)
        self.model = model_to_device(self.traced_model, self.device)

        # 关闭所有节点的连接
        self.close_child_nodes()