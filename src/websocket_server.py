from syft.workers.websocket_server import WebsocketServerWorker
import time
import torch
from src.websocket_client import MyWebsocketClientWorker
import syft as sy
from src.nn_model import ConvNet1D, loss_fn, model_to_device
from src import my_utils
from typing import Dict
from src.train_config import MyTrainConfig
from syft.workers.abstract import AbstractWorker


class MyWebsocketServerWorker(WebsocketServerWorker):

    def __init__(self, hook, host: str, port: int, id, verbose, owner: AbstractWorker = None):
        super().__init__(hook=hook, host=host, port=port, id=id, verbose=verbose)
        self.owner = self.owner = owner if owner else sy.hook.local_worker
        self.child_nodes: Dict[MyWebsocketClientWorker] = []

        # 训练部分的参数
        self.model = None
        self.traced_model = None
        self.loss_fn = loss_fn
        self.batch_size = 32
        self.learning_rate = 0.001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        # return self.model, self.traced_model

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
        train_config = MyTrainConfig(
            model=self.model,
            loss_fn=self.loss_fn,
            batch_size=self.batch_size,
            shuffle=True,
            # max_nr_batches=max_nr_batches,
            epochs=1,
            optimizer="SGD",
            optimizer_args={"lr": self.learning_rate},
        )
        for child_node in self.child_nodes:
            train_config.send(child_node)

        # 关闭所有节点的连接
        self.close_child_nodes()

    def model_collection(self, forward_device_mapping_id: dict, port=9292, model_id=int(str('1' * 11))):
        # 连接所有子节点
        self.connect_child_nodes(forward_device_mapping_id, port)

        # 收回所有模型
        for child_node in self.child_nodes:
            self.owner.request_obj(model_id, child_node)

        # 关闭所有节点的连接
        self.close_child_nodes()

    def train(self, dataset_key: str):

        self._check_train_config()
        print(f'{self.device} is available.')

        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        traced_model = self.get_obj(self.train_config._model_id).obj
        self.model = model_to_device(traced_model, self.device)
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        self._build_optimizer(
            self.train_config.optimizer, self.model, optimizer_args=self.train_config.optimizer_args
        )

        return self._train(dataset_key, loss_fn)

    def _train(self, dataset_key, loss_fn, model_id=int(str('1' * 11))):

        self.model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None
        iteration_count = 0

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

        self.model.eval()
        self.model = model_to_device(self.model, 'cpu')
        self.traced_model = torch.jit.trace(
            self.model,
            torch.zeros([1, 400, 3], dtype=torch.float).to('cpu')
        )

        self.register_obj(obj=self.traced_model, obj_id=model_id)

        return loss.to('cpu')
