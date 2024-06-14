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


async def send_command(commands, nodes):
    await asyncio.gather(
        *[
            n.async_command(cmd)
            for n, cmd in zip(nodes, commands)
        ]
    )


async def main():
    hook = sy.TorchHook(torch)
    train_config = Config()

    # 连接并测试所有节点
    all_nodes_id = ['AA', 'BB', 'CC', 'DD', 'EE']
    all_nodes = {}
    for node_id in all_nodes_id:
        all_nodes[node_id] = MyWebsocketClientWorker(hook=hook, **generate_kwarg(node_id))

    # 清除并断开连接
    for node in all_nodes.values():
        node.clear_objects_remote()
        node.close()

    # 在第一个设备初始化模型
    all_nodes['AA'].connect()
    all_nodes['AA'].command(generate_command_dict(command_name="model_initialization"))
    all_nodes['AA'].close()

    for cur_round in range(1, train_config.training_rounds+1):
        logging.info(f"Training round {cur_round}/{train_config.training_rounds}")

        # 下发模型
        all_nodes['AA'].connect()
        all_nodes['AA'].command(generate_command_dict(
            command_name="model_dissemination",
            forward_device_id=['BB', 'CC']
        ))
        all_nodes['AA'].close()

        all_nodes['BB'].connect()
        all_nodes['CC'].connect()
        cmds = [
            generate_command_dict(command_name="model_dissemination", forward_device_id=['DD']),
            generate_command_dict(command_name="model_dissemination", forward_device_id=['EE'])
        ]
        await send_command(commands=cmds, nodes=[all_nodes['BB'], all_nodes['CC']])
        all_nodes['BB'].close()
        all_nodes['CC'].close()

        # 开始进行训练
        for node in all_nodes.values():
            node.connect()
        await send_command(
            commands=[generate_command_dict(command_name="train", dateset_ket="HAR-1")],
            nodes=list(all_nodes.values())
        )
        for node in all_nodes.values():
            node.close()

        # 模型回收
        all_nodes['DD'].connect()
        all_nodes['EE'].connect()
        cmds = [
            generate_command_dict(command_name="model_collection", forward_device_id=['BB'], aggregation=True),
            generate_command_dict(command_name="model_collection", forward_device_id=['CC'], aggregation=True),
        ]
        await send_command(
            commands=cmds,
            nodes=[all_nodes['DD'], all_nodes['EE']]
        )
        all_nodes['DD'].close()
        all_nodes['EE'].close()

        all_nodes['BB'].connect()
        all_nodes['CC'].connect()
        cmds = [
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=False),
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=True),
        ]
        for cmd, n in zip(cmds, [all_nodes['BB'], all_nodes['CC']]):
            n.command(cmd)


if __name__ == '__main__':
    LOG_INTERVAL = 25
    logger = logging.getLogger("DFL Training")

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)