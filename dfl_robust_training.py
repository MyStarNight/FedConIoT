from dfl_training import *
from src import topology
from p_tree import Nodes
from src.config import Config
from src.websocket_client import MyWebsocketClientWorker
from src.my_utils import generate_kwarg, generate_command_dict

import syft as sy
import torch
import numpy as np
import random
import os
from datetime import datetime
from typing import List

nodes_id = ['AA', 'BB', 'CC', 'DD', 'EE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
agg_num = 3


def fault_detection(edf: Nodes):
    fault_node_index = 0
    for i, node in enumerate(edf.all_nodes):
        print(node.id)
        try:
            node.connect()
        except Exception as e:
            print(f"{node.id} failed: {e}")
            fault_node_index = i + 1
            break

    edf.robust_settings(fault_node_index)
    close_connection(edf.all_nodes)
    edf.find_best_policies()


async def robust_main(new_start=True, training_rounds=100, fault_nodes_id=None):
    hook = sy.TorchHook(torch)
    me = sy.hook.local_worker
    train_config = Config(training_rounds=training_rounds)

    save_path = "result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    # 配置训练节点参数
    edf = Nodes()
    edf.initialized_settings(nodes_id=nodes_id.copy(), agg_num=agg_num, seed=15)
    if fault_nodes_id is not None:
        for fault_node_id in fault_nodes_id:
            fault_node_index = edf.nodes_id.index(fault_node_id) + 1
            edf.robust_settings(fault_node_index)
    edf.find_best_policies()

    pull_time = []
    push_time = []
    train_time = []
    accuracy_list = []

    for node_id in edf.nodes_id:
        edf.all_nodes.append(MyWebsocketClientWorker(hook=hook, **generate_kwarg(node_id)))
    close_connection(edf.all_nodes)

    # 选择节点初始化模型
    if new_start:
        initialized_model(edf.all_nodes[edf.agg[0]])
    else:
        keep_training_model(edf.all_nodes[edf.agg[0]])

    # 开始训练
    cur_round = 1
    while True:
        if cur_round > train_config.training_rounds:
            break
        logger.info(f"Training round {cur_round}/{train_config.training_rounds}")

        index = (cur_round-1) % len(edf.agg)
        agg_index = cur_round % len(edf.agg)
        node_pull_tree = edf.node_pull_trees[index]
        node_push_tree = edf.node_push_trees[index]

        # 下发模型
        logger.info(f"model dissemination")
        start = datetime.now()
        await disseminate_model(node_pull_tree, edf.nodes_id, edf.all_nodes)
        pull_time.append((datetime.now() - start).total_seconds())

        # 调整发送模型的训练状态
        change_state(edf.all_nodes[edf.agg[index]])

        # 训练模型
        logger.info("model training")
        start = datetime.now()
        await train_model(edf.all_nodes)
        train_time.append((datetime.now() - start).total_seconds())

        # 模型回收
        logger.info("model collection")
        start = datetime.now()
        await collect_model(node_push_tree, edf.nodes_id, edf.all_nodes)
        push_time.append((datetime.now() - start).total_seconds())

        if cur_round % 5 == 0 or cur_round == train_config.training_rounds:
            model = set_federated_model(edf.all_nodes[edf.agg[agg_index]], me)
            accuracy = evaluate(model)
            accuracy_list.append(accuracy)

            # 保存所需的数据
            df_time = pd.DataFrame([pull_time, train_time, push_time], index=['pull', 'train', 'push']).T
            df_accuracy = pd.DataFrame(accuracy_list, index=[5 * (i + 1) for i in range(len(accuracy_list))])

            df_time.to_csv(f'{save_path}/time_{time_str}.csv')
            df_accuracy.to_csv(f'{save_path}/accuracy_{time_str}.csv')
            torch.save(model.state_dict(), 'result/model.pth')

        cur_round = cur_round + 1

    return pull_time, train_time, push_time, accuracy_list

if __name__ == '__main__':
    LOG_INTERVAL = 25
    logger = logging.getLogger("DFL Training")

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    n_s = True
    t_r = 10
    fn_id = None

    pull_time, train_time, push_time, accuracy_list =\
        asyncio.get_event_loop().run_until_complete(robust_main(
            new_start=n_s,
            training_rounds=t_r,
            fault_nodes_id=fn_id
        ))

    visualization(accuracy_list)
