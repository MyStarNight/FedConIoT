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
import dfl_training as dfl
import pandas as pd
import os


async def send_command(commands, nodes):
    await asyncio.gather(
        *[
            n.async_command(cmd)
            for n, cmd in zip(nodes, commands)
        ]
    )


def clear_central_node_storage(node: MyWebsocketClientWorker):
    dfl.start_connection([node])
    node.command(generate_command_dict(command_name="central_node_storage_clear"))
    dfl.close_connection([node])


async def main():
    hook = sy.TorchHook(torch)
    me = sy.hook.local_worker
    train_config = Config(training_rounds=100)

    pull_time = []
    push_time = []
    train_time = []
    accuracy_list = []

    all_nodes_id = ['testing', 'AA', 'BB', 'CC', 'EE', 'DD']
    all_nodes = []
    for node_id in all_nodes_id:
        all_nodes.append(MyWebsocketClientWorker(hook=hook, **generate_kwarg(node_id)))
    dfl.close_connection(all_nodes)

    node_pull_tree = {1: [(0, i) for i in range(1, len(all_nodes))]}
    node_push_tree = {1: [(i, 0) for i in range(1, len(all_nodes))]}

    dfl.initialized_model(all_nodes[0])

    for cur_round in range(1, train_config.training_rounds + 1):
        logger.info(f"Training round {cur_round}/{train_config.training_rounds}")

        # 下发模型
        logger.info(f"model dissemination")
        start = datetime.now()
        await dfl.disseminate_model(node_pull_tree, all_nodes_id, all_nodes)
        pull_time.append((datetime.now() - start).total_seconds())

        # 训练模型
        logger.info("model training")
        start = datetime.now()
        await dfl.train_model(all_nodes[1:])
        train_time.append((datetime.now() - start).total_seconds())

        # 清空
        clear_central_node_storage(all_nodes[0])

        # 模型回收
        logger.info("model collection")
        start = datetime.now()
        await dfl.collect_model(node_push_tree, all_nodes_id, all_nodes)
        push_time.append((datetime.now() - start).total_seconds())

        if cur_round % 5 == 0 or cur_round == train_config.training_rounds:
            model = dfl.set_federated_model(all_nodes[0], me)
            accuracy = evaluate(model)
            accuracy_list.append(accuracy)

    return pull_time, train_time, push_time, accuracy_list


if __name__ == '__main__':
    LOG_INTERVAL = 25
    logger = logging.getLogger("DFL Training")

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    pull_time, train_time, push_time, accuracy_list = asyncio.get_event_loop().run_until_complete(main())

    df_time = pd.DataFrame([pull_time, train_time, push_time], index=['pull', 'train', 'push']).T
    df_accuracy = pd.DataFrame(accuracy_list, index=[5*(i+1) for i in range(len(accuracy_list))])

    save_path = "result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    df_time.to_csv(f'{save_path}/time_{time_str}.csv')
    df_accuracy.to_csv(f'{save_path}/accuracy_{time_str}.csv')



