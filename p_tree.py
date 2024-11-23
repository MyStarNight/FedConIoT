from src import topology
import numpy as np


def change_index(input_dict):
    new_dict = {}

    for key, tuple_list in input_dict.items():
        new_dict[key] = [(x - 1, y - 1) for x, y in tuple_list]

    return new_dict


class Nodes:
    def __init__(self):
        self.agg_num = None

        self.nodes_id = None
        self.adj_matrix = None
        self.t_sum_list = None
        self.node_pull_result = None
        self.node_push_result = None

        self.agg = None
        self.node_pull_trees = None
        self.node_push_trees = None

        self.all_nodes = []

    def initialized_settings(self, nodes_id, agg_num, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.nodes_id = nodes_id
        self.agg_num = agg_num
        self.adj_matrix = topology.generate_matrix(len(nodes_id))
        self.node_pull_result, self.node_push_result, self.t_sum_list = \
            topology.topology_to_policy(matrix=self.adj_matrix)

    def robust_settings(self, failed_nodes_index):

        # 首先删除不可使用的节点
        self.nodes_id.pop(failed_nodes_index-1)
        # self.all_nodes.pop(failed_nodes_index-1)
        self.delete_failed_node(failed_nodes_index)
        self.node_pull_result, self.node_push_result, self.t_sum_list = \
            topology.topology_to_policy(matrix=self.adj_matrix)

    def find_best_policies(self):
        # 选择合适的聚合节点进行聚合
        sorted_indices = np.argsort(self.t_sum_list) + 1
        agg_indices = list(sorted_indices[:self.agg_num])

        # 选定回收的顺序
        back_agg_indices = agg_indices[1:]
        back_agg_indices.append(agg_indices[0])

        # 选择模型下发路径
        node_pull_trees, node_push_trees = [], []
        for i in range(len(agg_indices)):
            node_pull_trees.append(change_index(self.node_pull_result[agg_indices[i]][1]))
            node_push_trees.append(change_index(self.node_push_result[back_agg_indices[i]][1]))

        self.agg = [i-1 for i in agg_indices]
        self.node_pull_trees = node_pull_trees
        self.node_push_trees = node_push_trees

    def delete_failed_node(self, failed_node_index):
        k = failed_node_index - 1
        self.adj_matrix = np.delete(self.adj_matrix, k, axis=0)
        self.adj_matrix = np.delete(self.adj_matrix, k, axis=1)



class Nodes5:
    def __init__(self):
        self.j_5 = ['AA', 'BB', 'CC', 'DD', 'EE']
        self.r_5 = ['A', 'B', 'C', 'D', 'E']
        self.agg = [i-1 for i in [1, 3]]

        self.node_pull_tree_1 = change_index({1: [(1, 2), (1, 3)], 2: [(2, 5), (3, 4)]})
        self.node_push_tree_1 = change_index({1: [(4, 2), (5, 1)], 2: [(2, 3), (1, 3)]})

        self.node_pull_tree_2 = change_index({1: [(3, 2), (3, 4)], 2: [(2, 1), (4, 5)]})
        self.node_push_tree_2 = change_index({1: [(3, 2), (4, 5)], 2: [(2, 1), (5, 1)]})

        self.node_pull_trees = [self.node_pull_tree_1, self.node_pull_tree_2]
        self.node_push_trees = [self.node_push_tree_1, self.node_push_tree_2]


class Nodes7:
    def __init__(self):
        self.j_3_r_4 = ["AA", "BB", "A", "B", "C", "CC", "D"]
        self.j_5_r_2 = ["AA", "A", "BB", "CC", "DD", "EE", "B"]
        self.r_7 = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.agg = [0, 5]
        self.node_pull_tree_1 = change_index({1: [(1, 2), (1, 5), (1, 4)], 2: [(5, 3), (2, 6), (4, 7)]})
        self.node_push_tree_1 = change_index({1: [(1, 2), (4, 7), (5, 3)], 2: [(2, 6), (3, 6), (7, 6)]})

        self.node_pull_tree_2 = change_index({1: [(6, 1), (6, 3), (6, 5)], 2: [(3, 2), (1, 4), (5, 7)]})
        self.node_push_tree_2 = change_index({1: [(3, 6), (5, 2), (7, 4)], 2: [(2, 1), (6, 1), (4, 1)]})

        self.node_pull_trees = [self.node_pull_tree_1, self.node_pull_tree_2]
        self.node_push_trees = [self.node_push_tree_1, self.node_push_tree_2]


class Nodes10:
    def __init__(self):
        self.j_5_r_5 = ['A', 'AA', 'B', 'BB', 'C', 'CC', 'D', 'DD', 'E', 'EE']
        self.r_10 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.agg = [1, 3, 5]
        self.node_pull_tree_1 = change_index(
            {1: [(2, 1), (2, 7), (2, 5), (2, 8), (2, 10)], 2: [(7, 3), (5, 4), (8, 6), (10, 9)]})
        self.node_push_tree_1 = change_index(
            {1: [(1, 9), (2, 5), (8, 6), (10, 7)], 2: [(9, 4), (5, 4), (3, 4), (6, 4), (7, 4)]})

        self.node_pull_tree_2 = change_index(
            {1: [(4, 10), (4, 7), (4, 8)], 2: [(10, 1), (7, 2), (7, 3), (8, 5), (8, 6), (10, 9)]})
        self.node_push_tree_2 = change_index(
            {1: [(2, 8), (3, 5), (4, 8), (7, 5), (9, 1), (10, 1)], 2: [(1, 6), (8, 6), (5, 6)]})

        self.node_pull_tree_3 = change_index(
            {1: [(6, 1), (6, 9), (6, 5), (6, 4), (6, 8)], 2: [(9, 2), (5, 3), (8, 7), (4, 10)]})
        self.node_push_tree_3 = change_index(
            {1: [(1, 3), (4, 7), (5, 3), (6, 9), (8, 7), (10, 9)], 2: [(3, 2), (7, 2), (9, 2)]})

        self.node_pull_trees = [self.node_pull_tree_1, self.node_pull_tree_2, self.node_pull_tree_3]
        self.node_push_trees = [self.node_push_tree_1, self.node_push_tree_2, self.node_push_tree_3]


class Nodes13:
    def __init__(self):
        self.j_3_r_10 = ['AA', 'BB', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'CC']
        self.j_5_r_8 = ['AA', 'BB', 'A', 'B', 'C', 'DD', 'EE', 'F', 'G', 'H', 'I', 'J', 'CC']
        self.agg = [i - 1 for i in [1, 2, 6, 13]]
        self.node_pull_tree_1 = change_index({1: [(1, 4), (1, 11), (1, 10), (1, 7), (1, 8)],
                                              2: [(4, 2), (11, 3), (10, 5), (10, 6), (11, 9), (8, 12), (7, 13)]})
        self.node_push_tree_1 = change_index({1: [(1, 4), (3, 6), (7, 12), (8, 12), (10, 6), (11, 9), (13, 9)],
                                              2: [(4, 2), (6, 2), (5, 2), (12, 2), (9, 2)]})

        self.node_pull_tree_2 = change_index({1: [(2, 9), (2, 3), (2, 4), (2, 10), (2, 6)],
                                              2: [(9, 1), (10, 5), (4, 7), (4, 8), (6, 11), (10, 12), (9, 13)]})
        self.node_push_tree_2 = change_index({1: [(1, 10), (4, 7), (5, 2), (8, 7), (9, 3), (11, 3), (12, 2), (13, 10)],
                                              2: [(10, 6), (2, 6), (3, 6), (7, 6)]})

        self.node_pull_tree_3 = change_index({1: [(6, 1), (6, 2), (6, 10), (6, 12), (6, 11), (6, 9)],
                                              2: [(10, 3), (2, 4), (12, 5), (11, 7), (1, 8), (9, 13)]})
        self.node_push_tree_3 = change_index({1: [(1, 11), (2, 4), (6, 9), (8, 7), (10, 3), (12, 5)],
                                              2: [(11, 13), (4, 13), (3, 13), (5, 13), (9, 13), (7, 13)]})

        self.node_pull_tree_4 = change_index({1: [(13, 1), (13, 9), (13, 10), (13, 8)],
                                              2: [(9, 2), (10, 3), (9, 4), (8, 5), (10, 6), (1, 7), (1, 11), (8, 12)]})
        self.node_push_tree_4 = change_index({1: [(2, 9), (3, 6), (5, 13), (7, 6), (8, 9), (10, 4), (11, 13), (12, 4)],
                                              2: [(9, 1), (6, 1), (4, 1), (13, 1)]})

        self.node_pull_trees = [
            self.node_pull_tree_1,
            self.node_pull_tree_2,
            self.node_pull_tree_3,
            self.node_pull_tree_4
        ]

        self.node_push_trees = [
            self.node_push_tree_1,
            self.node_push_tree_2,
            self.node_push_tree_3,
            self.node_push_tree_4
        ]


class Nodes15:
    def __init__(self):
        self.j_5_r_10 = ['AA', 'BB', 'CC', 'DD', 'EE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.agg = [i - 1 for i in [1, 2, 3, 4]]
        self.node_pull_tree_1 = change_index({1: [(1, 13), (1, 14), (1, 6), (1, 15), (1, 11)],
                                              2: [(13, 2),
                                                  (14, 3),
                                                  (6, 4),
                                                  (15, 5),
                                                  (15, 7),
                                                  (13, 8),
                                                  (14, 9),
                                                  (11, 10),
                                                  (11, 12)]})
        self.node_push_tree_1 = change_index({1: [(1, 13), (4, 12), (5, 11), (6, 10), (7, 9), (14, 3), (15, 8)],
                                              2: [(13, 2), (3, 2), (12, 2), (11, 2), (10, 2), (9, 2), (8, 2)]})

        self.node_pull_tree_2 = change_index({1: [(2, 11), (2, 10), (2, 15), (2, 6), (2, 8)],
                                              2: [(11, 1),
                                                  (11, 3),
                                                  (10, 4),
                                                  (15, 5),
                                                  (15, 7),
                                                  (8, 9),
                                                  (8, 12),
                                                  (6, 13),
                                                  (10, 14)]})
        self.node_push_tree_2 = change_index(
            {1: [(1, 14), (2, 11), (4, 11), (7, 14), (8, 9), (10, 5), (12, 9), (15, 5)],
             2: [(14, 3), (11, 3), (5, 3), (6, 3), (9, 3), (13, 3)]})

        self.node_pull_tree_3 = change_index({1: [(3, 6), (3, 2), (3, 5), (3, 14)],
                                              2: [(6, 1),
                                                  (6, 4),
                                                  (14, 7),
                                                  (5, 8),
                                                  (14, 9),
                                                  (2, 10),
                                                  (5, 11),
                                                  (6, 12),
                                                  (14, 13),
                                                  (5, 15)]})
        self.node_push_tree_3 = change_index({1: [(1, 6),
                                                  (2, 10),
                                                  (3, 6),
                                                  (5, 9),
                                                  (7, 9),
                                                  (11, 10),
                                                  (12, 10),
                                                  (13, 8),
                                                  (14, 9),
                                                  (15, 8)],
                                              2: [(6, 4), (10, 4), (9, 4), (8, 4)]})

        self.node_pull_tree_4 = change_index({1: [(4, 12), (4, 11), (4, 6), (4, 9)],
                                              2: [(12, 1), (12, 2), (11, 3), (11, 5), (9, 15), (6, 13), (9, 10),
                                                  (6, 14)],
                                              3: [(15, 7), (13, 8)]})
        self.node_push_tree_4 = change_index({1: [(9, 4)],
                                              2: [(2, 11),
                                                  (3, 6),
                                                  (4, 12),
                                                  (5, 11),
                                                  (7, 12),
                                                  (10, 11),
                                                  (13, 8),
                                                  (14, 6),
                                                  (15, 8)],
                                              3: [(11, 1), (6, 1), (12, 1), (8, 1)]})

        self.node_pull_trees = [
            self.node_pull_tree_1,
            self.node_pull_tree_2,
            self.node_pull_tree_3,
            self.node_pull_tree_4
        ]

        self.node_push_trees = [
            self.node_push_tree_1,
            self.node_push_tree_2,
            self.node_push_tree_3,
            self.node_push_tree_4
        ]


if __name__ == '__main__':
    # 测试代码
    nodes_id = ['AA', 'BB', 'CC', 'DD', 'EE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    edf = Nodes()
    edf.initialized_settings(nodes_id.copy(), 3, seed=15)
    edf.find_best_policies()
