from syft.workers.websocket_server import WebsocketServerWorker
import time
import torch
from my_utils.websocket_client import MyWebsocketClientWorker
import syft as sy


class MyWebsocketServerWorker(WebsocketServerWorker):

    def __init__(self, hook, host: str, port: int, id, verbose):
        super().__init__(hook=hook, host=host, port=port, id=id, verbose=verbose)
        self.owner = sy.local_worker
        self.child_nodes = []

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

        return response_list

    @staticmethod
    def test():
        print("test start")
        time.sleep(5)
        print("test end")
