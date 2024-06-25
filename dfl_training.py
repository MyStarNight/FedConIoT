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
from typing import List
from src import topology


async def send_command(commands, nodes):
    await asyncio.gather(
        *[
            n.async_command(cmd)
            for n, cmd in zip(nodes, commands)
        ]
    )


def start_connection(nodes: List[MyWebsocketClientWorker]):
    for node in nodes:
        node.connect()


def close_connection(nodes: List[MyWebsocketClientWorker]):
    for node in nodes:
        node.close()


def initialized_model(node: MyWebsocketClientWorker):
    start_connection([node])
    node.command(generate_command_dict(command_name="model_initialization"))
    close_connection([node])


def change_state(node: MyWebsocketClientWorker):
    start_connection([node])
    node.command(generate_command_dict(command_name="change_state"))
    close_connection([node])


async def disseminate_model(pull_tree: dict, all_nodes_id: list, all_nodes: list):
    for level in pull_tree.values():
        send_node_indices = set([node[0] for node in level])
        send_nodes = [all_nodes[index] for index in send_node_indices]
        start_connection(send_nodes)

        cmds = []
        for send_node_index in send_node_indices:
            receive_node_indices = [s for r, s in level if r == send_node_index]
            forward_device_id = [all_nodes_id[index] for index in receive_node_indices]
            command = generate_command_dict(
                command_name="model_dissemination",
                forward_device_id=forward_device_id
            )
            cmds.append(command)

        await send_command(commands=cmds, nodes=send_nodes)
        close_connection(send_nodes)


async def train_model(all_nodes: List[MyWebsocketClientWorker], test=False):
    start_connection(all_nodes)
    commands = [generate_command_dict(command_name="train", dataset_key="HAR-1", test=test)] * len(all_nodes)
    await send_command(
        commands=commands,
        nodes=all_nodes
    )
    close_connection(all_nodes)


def pull_level_to_dict(level: list):
    receive_node_indices = set([node[1] for node in level])
    level_dict = {index: [] for index in receive_node_indices}
    for r, s in level:
        level_dict[s].append(r)

    times = [len(value) for value in level_dict.values()]
    counts = max(times)

    return level_dict, counts


async def collect_model(push_tree: dict, all_nodes_id: list, all_nodes: list):
    for level in push_tree.values():
        level_dict, counts = pull_level_to_dict(level)
        for cnt in range(counts):
            send_node_indices = []
            cmds = []
            for r, s in level_dict.items():
                if len(s) < cnt + 1:
                    break
                send_node_indices.append(s[cnt])
                forward_device_id = [all_nodes_id[r]]
                aggregation = True if len(s) == cnt + 1 else False
                cmds.append(generate_command_dict(
                    command_name="model_collection", forward_device_id=forward_device_id, aggregation=aggregation))

            send_nodes = [all_nodes[index] for index in send_node_indices]
            start_connection(send_nodes)
            await send_command(commands=cmds, nodes=send_nodes)
            close_connection(send_nodes)


def set_federated_model(node: MyWebsocketClientWorker, me):
    start_connection([node])
    node.command(generate_command_dict(command_name="set_federated_model"))
    model = me.request_obj(int(str(1) * 11), node).obj
    close_connection([node])
    return model


async def main():
    hook = sy.TorchHook(torch)
    me = sy.hook.local_worker
    train_config = Config(training_rounds=50)

    # node_pull_tree = {1: [(1, 0), (1, 2), (1, 3)], 2: [(0, 4), (2, 5), (3, 6)]}
    # node_push_tree = {1: [(4, 0), (5, 2), (6, 3)], 2: [(0, 1), (2, 1), (3, 1)]}

    node_pull_tree = {1: [(1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]}
    node_push_tree = {1: [(4, 1), (5, 1), (6, 1), (0, 1), (2, 1), (3, 1)]}

    pull_time = []
    push_time = []
    train_time = []

    # 指定节点并进行存储
    all_nodes_id = ['AA', 'BB', 'CC', 'DD', 'E', 'F', 'G']
    all_nodes = []
    for node_id in all_nodes_id:
        all_nodes.append(MyWebsocketClientWorker(hook=hook, **generate_kwarg(node_id)))
    close_connection(all_nodes)

    # 选择节点初始化模型
    initialized_model(all_nodes[2-1])

    # 开始训练
    for cur_round in range(1, train_config.training_rounds+1):
        logger.info(f"Training round {cur_round}/{train_config.training_rounds}")

        # 下发模型
        logger.info(f"model dissemination")
        start = datetime.now()
        await disseminate_model(node_pull_tree, all_nodes_id, all_nodes)
        pull_time.append((datetime.now() - start).total_seconds())

        # 调整发送模型的训练状态
        change_state(all_nodes[2 - 1])

        # 训练模型
        logger.info("model training")
        start = datetime.now()
        await train_model(all_nodes)
        train_time.append((datetime.now() - start).total_seconds())

        # 模型回收
        logger.info("model collection")
        start = datetime.now()
        await collect_model(node_push_tree, all_nodes_id, all_nodes)
        push_time.append((datetime.now() - start).total_seconds())

        if cur_round % 5 == 0 or cur_round == train_config.training_rounds:
            model = set_federated_model(all_nodes[2-1], me)
            evaluate(model)

    return pull_time, train_time, push_time


if __name__ == '__main__':
    LOG_INTERVAL = 25
    logger = logging.getLogger("DFL Training")

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    pull_time, train_time, push_time = asyncio.get_event_loop().run_until_complete(main())