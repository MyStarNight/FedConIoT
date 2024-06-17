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


async def send_command(commands, nodes):
    await asyncio.gather(
        *[
            n.async_command(cmd)
            for n, cmd in zip(nodes, commands)
        ]
    )


async def main():
    hook = sy.TorchHook(torch)
    me = sy.hook.local_worker
    train_config = Config(training_rounds=20)

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

    pull_time = []
    push_time = []
    train_time = []

    for cur_round in range(1, train_config.training_rounds+1):
        logger.info(f"Training round {cur_round}/{train_config.training_rounds}")

        start = datetime.now()
        # 下发模型
        logger.info(f"model dissemination")
        all_nodes['AA'].connect()
        all_nodes['AA'].command(generate_command_dict(command_name="change_state"))
        all_nodes['AA'].command(generate_command_dict(
            command_name="model_dissemination",
            forward_device_id=['BB', 'CC', 'DD', 'EE']
        ))
        all_nodes['AA'].close()
        pull_time.append((datetime.now()-start).total_seconds())

        start = datetime.now()
        # 开始进行训练
        logger.info("model training")
        for node in all_nodes.values():
            node.connect()
        await send_command(
            commands=[generate_command_dict(command_name="train", dataset_key="HAR-1")]*5,
            nodes=list(all_nodes.values())
        )
        for node in all_nodes.values():
            node.close()
        train_time.append((datetime.now() - start).total_seconds())

        start = datetime.now()
        # 模型回收
        logger.info("model collection")
        all_nodes['DD'].connect()
        all_nodes['EE'].connect()
        all_nodes['BB'].connect()
        all_nodes['CC'].connect()

        cmds = [
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=False),
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=False),
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=False),
            generate_command_dict(command_name="model_collection", forward_device_id=['AA'], aggregation=True),
        ]
        for cmd, n in zip(cmds, list(all_nodes.values())[1:]):
            n.command(cmd)

        all_nodes['DD'].close()
        all_nodes['EE'].close()
        all_nodes['BB'].close()
        all_nodes['CC'].close()

        push_time.append((datetime.now() - start).total_seconds())

        if cur_round % 1 == 0 or cur_round == train_config.training_rounds:
            all_nodes['AA'].connect()
            all_nodes['AA'].command(generate_command_dict(command_name="set_federated_model"))
            model = me.request_obj(int(str(1)*11), all_nodes['AA']).obj
            all_nodes['AA'].close()
            evaluate(model)

    return pull_time, train_time, push_time


if __name__ == '__main__':
    LOG_INTERVAL = 25
    logger = logging.getLogger("DFL Training")

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    pull_time, train_time, push_time = asyncio.get_event_loop().run_until_complete(main())
