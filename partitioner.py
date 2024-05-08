from __future__ import annotations
import copy
from functools import cmp_to_key
import sys
import threading
import uuid
import json
import time
import networkx as nx
import matplotlib.pyplot as plt

MEMORY_DB_READ_LATENCY = 0.3 / 1000
MEMORY_DB_WRITE_LATENCY = 5 / 1000
DYNAMO_DB_READ_LATENCY = 10 / 1000
DYNAMO_DB_WRITE_LATENCY = 10 / 1000

dynamodb = {}
memory_db = {}

lock = threading.Lock()

threads = set()

cost_d_per_gb = 0
cost_m_per_gb = 0.2


def datastore_get(checkpoint_name):
    time.sleep(DYNAMO_DB_READ_LATENCY)
    return dynamodb.get(checkpoint_name)


def datastore_atomic_add(checkpoint_name, result):
    time.sleep(DYNAMO_DB_READ_LATENCY)
    if checkpoint_name in dynamodb:
        return False
    with lock:
        time.sleep(DYNAMO_DB_WRITE_LATENCY)
        dynamodb[checkpoint_name] = result
        return True


def datastore_atomic_update_set(set_id, checkpoint_name):
    with lock:
        dynamodb[set_id]["values"].add(checkpoint_name)
        return True


def async_invoke(node: Node, result):
    t = threading.Thread(target=node.ingress, args=(result,))
    t.start()
    threads.add(t)
    return t


def add(args):
    a, b = args
    time.sleep(0.01)
    return a + b


def double(number):
    time.sleep(0.02)
    return number * 2


def exp(e):
    def wrapper(a):
        time.sleep(0.03)
        return a**e

    return wrapper


HANDLES = {
    "add": {"handler": add, "time": 0.01},
    "double": {"handler": double, "time": 0.02},
    "exp": {"handler": exp, "time": 0.03},
}


def byte_length(i):
    return (i.bit_length() + 7) // 8


class Node:
    def __init__(self, handle, exec_time, nid=None) -> None:
        self.id = nid or str(uuid.uuid4())
        self.in_degrees: list[Edge] = []
        self.out_degrees: list[Edge] = []
        self.handle = handle
        self.checkpoint_name = str(uuid.uuid4())
        self.checkpoint_sets = {}
        self.cv = threading.Condition()
        self.executed = False
        self.priority = None
        self.rank_downward = None
        self.rank_upward = None
        self.exec_time = exec_time
        self.is_critical = False
        self.use_memory_store = False

    @property
    def required_bytes(self) -> int:
        """Return required bytes for storing result and checkpoint"""
        return byte_length(0) * len(self.in_degrees) + len(
            json.dumps(
                {
                    "values": [0 for _ in self.in_degrees],
                    "size": len(self.in_degrees),
                },
                indent=2,
            ).encode("utf-8")
        )

    @property
    def parents(self) -> list[Node]:
        return [in_degree.src for in_degree in self.in_degrees]

    @property
    def children(self) -> list[Node]:
        return [out_degree.dst for out_degree in self.out_degrees]

    @property
    def next_functions(self):
        return [child for child in self.children]

    def wait(self):
        with self.cv:
            self.cv.wait_for(lambda: self.executed)

    def ingress(self, input):
        result = datastore_get(self.checkpoint_name)
        if result:
            self._egress(result)
        else:
            self.egress(self.handle(input))

    def egress(self, result):
        if not datastore_atomic_add(self.checkpoint_name, result):
            result = datastore_get(self.checkpoint_name)
        else:
            for _, checkpoint_set in self.checkpoint_sets.items():
                datastore_atomic_update_set(
                    checkpoint_set, (self.checkpoint_name, result)
                )
        self._egress(result)

    def _egress(self, result):
        for f in self.next_functions:
            if f.id in self.checkpoint_sets:
                set_result = datastore_get(self.checkpoint_sets[f.id])
                if len(set_result["values"]) != set_result["size"]:
                    continue
                else:
                    result = sum(value[1] for value in set_result["values"])
            async_invoke(f, result)
        self.executed = True
        with self.cv:
            self.cv.notify_all()


class Edge:
    src: Node
    dst: Node
    data_size: int

    def __init__(self, src, dst, size=1) -> None:
        self.src = src
        self.dst = dst
        self.data_size = size


class Graph:
    def __init__(self, filepath) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            conf = json.load(f)

        self.conf = conf
        self.start_node: Node = None
        self.end_node: Node = None
        self.graph = nx.Graph()
        self.position: dict[int, list[int]] = {}
        self.nodes: set[Node] = set()

        for node_conf in conf.get("Nodes", []):
            if node_conf.get("Start"):
                self._setup_node(node_conf)
                break
        dfs(self.start_node, set())
        self._find_critical_path()

    def _calculate_rank_upward(self, node: Node):
        if node.rank_upward:
            return node.rank_upward
        max_cost = 0
        for edge in node.out_degrees:
            cost = edge.data_size * len(
                edge.dst.in_degrees
            ) + +self._calculate_rank_upward(edge.dst)
            if cost > max_cost:
                max_cost = cost
        node.rank_upward = node.exec_time + max_cost
        return node.rank_upward

    def _calculate_rank_downward(self, node: Node):
        if node.rank_downward:
            return node.rank_downward
        max_cost = 0
        for edge in node.in_degrees:
            cost = (
                edge.data_size
                + edge.src.exec_time
                + self._calculate_rank_downward(edge.src)
            )
            if cost > max_cost:
                max_cost = cost
        node.rank_downward = max_cost
        return node.rank_downward

    def _set_critical_node(self, node: Node):
        node.is_critical = True

        critical_id = 0
        max_priority = 0
        for i, out_edge in enumerate(node.out_degrees):
            if out_edge.dst.priority > max_priority:
                max_priority = out_edge.dst.priority
                critical_id = i
        if node.out_degrees:
            self._set_critical_node(node.out_degrees[critical_id].dst)

    def _find_critical_path(self):
        self._calculate_rank_downward(self.end_node)
        self._calculate_rank_upward(self.start_node)
        for node in self.nodes:
            node.priority = node.rank_upward + node.rank_downward
        self._set_critical_node(self.start_node)

    def _setup_node(self, node_conf, level=0):
        for node in self.nodes:
            if node.id == node_conf["Id"]:
                return node
        handle = HANDLES[node_conf["Op"]]["handler"]
        exc_time = HANDLES[node_conf["Op"]]["time"]
        if "Args" in node_conf:
            handle = handle(*node_conf["Args"])
        node = Node(handle, exc_time, nid=node_conf["Id"])
        self.nodes.add(node)
        self.graph.add_node(node.id)
        if level not in self.position:
            self.position[level] = []

        self.position[level].append(node.id)
        if node_conf.get("Start"):
            self.start_node = node
        elif node_conf.get("End"):
            self.end_node = node

        children_conf = node_conf["Children"]
        for child_id in children_conf:
            for _node_conf in self.conf["Nodes"]:
                if _node_conf["Id"] == child_id:
                    child = self._setup_node(_node_conf, level + 1)
                    connect(node, child)
                    self.graph.add_edge(node.id, child.id)
                    break
        return node

    def async_start(self, args):
        async_invoke(self.start_node, args)

    def wait(self):
        self.end_node.wait()

    def result(self):
        return datastore_get(self.end_node.checkpoint_name)

    def print(self):
        position = {}
        for level, ids in self.position.items():
            for i, nid in enumerate(ids):
                position[nid] = (1 + 5 * i, -3 * (level + 1))
        nx.draw_networkx(self.graph, arrows=True, pos=position)
        plt.savefig("test.png", format="PNG")
        plt.close()

    def print_nodes(self):
        queue: list[Node] = []
        queue.append(self.start_node)

        visited = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if node.id in visited:
                continue
            visited.add(node.id)
            print(node.id, node.is_critical, datastore_get(node.checkpoint_name))
            for child in node.children:
                queue.append(child)

    def _get_stats(self, nodes):
        e2e_latency = 0
        checkpoint_time = 0
        additional_cost = 0

        for node in nodes:
            if node.use_memory_store:
                additional_cost += (
                    cost_m_per_gb * node.required_bytes / 1024 / 1024 / 1024
                )
                memorydb_latency = (
                    MEMORY_DB_WRITE_LATENCY
                    + MEMORY_DB_READ_LATENCY * len(node.in_degrees)
                )
                e2e_latency += memorydb_latency
                checkpoint_time += memorydb_latency
            else:
                dynamodb_latency = (
                    DYNAMO_DB_WRITE_LATENCY
                    + DYNAMO_DB_READ_LATENCY * len(node.in_degrees)
                )
                e2e_latency += dynamodb_latency
                checkpoint_time += dynamodb_latency
            e2e_latency += node.exec_time

        print(f"{e2e_latency=}, {checkpoint_time=}, {additional_cost=}")
        return e2e_latency, checkpoint_time, additional_cost

    def stats(self):
        """Return approximate end-to-end latency and checkpointing time1"""
        critical_nodes = self.get_critical_path()
        return self._get_stats(critical_nodes)

    def get_critical_path(self) -> list[Node]:
        critical_nodes: list[Node] = []
        queue: list[Node] = [self.start_node]
        while len(queue) > 0:
            node = queue.pop(0)
            critical_nodes.append(node)
            for child in node.children:
                if child.is_critical:
                    queue.append(child)
                    break
        return critical_nodes

    # def partition_by_bf(self, slo):
    #     """Partition by brute force"""
    #     critical_nodes = self.get_critical_path()

    #     sorted_by_needs: list[Node] = sorted(
    #         critical_nodes,
    #         key=cmp_to_key(
    #             lambda node1, node2: node2.required_bytes - node1.required_bytes
    #         ),
    #     )
    #     selected_nodes, _, _, _ = self._select_nodes(sorted_by_needs, slo)

    #     for node in selected_nodes:
    #         node.use_memory_store = True

    def partition_by_dp(self, slo):
        """Partition by dynamic programming"""
        critical_nodes = self.get_critical_path()
        cache: dict[frozenset, tuple[float, float, float]] = {}

        sorted_by_needs: list[Node] = sorted(
            critical_nodes,
            key=cmp_to_key(
                lambda node1, node2: node2.required_bytes - node1.required_bytes
            ),
        )
        selected_nodes, _, _, _ = self._select_nodes(sorted_by_needs, slo, cache)

        for node in selected_nodes:
            node.use_memory_store = True

    def _select_nodes(
        self, nodes: list[Node], slo, cache
    ) -> tuple[list[Node], float, float, float]:
        if len(nodes) == 0:
            return [], 0, 0, 0

        key = frozenset(node.id for node in nodes)
        if key in cache:
            return cache[key]
        min_latency = None
        min_ckpt_time = None
        min_nodes = None
        min_cost = None
        for i, node in enumerate(nodes):
            subnodes = nodes[:i] + nodes[i + 1 :]
            selected_nodes, latency, ckpt_time, cost = self._select_nodes(
                subnodes, slo, cache
            )

            dynamodb_latency = DYNAMO_DB_WRITE_LATENCY + DYNAMO_DB_READ_LATENCY * len(
                node.in_degrees
            )
            new_latency = latency + dynamodb_latency + node.exec_time
            new_ckpt_time = ckpt_time + dynamodb_latency

            if new_ckpt_time / new_latency > slo:
                cost += cost_m_per_gb * node.required_bytes / 1024 / 1024 / 1024
                memorydb_latency = (
                    MEMORY_DB_WRITE_LATENCY
                    + MEMORY_DB_READ_LATENCY * len(node.in_degrees)
                )
                new_latency = latency + memorydb_latency + node.exec_time
                new_ckpt_time = ckpt_time + memorydb_latency
                selected_nodes.append(node)
                selected_nodes = list(set(selected_nodes))

            if min_cost is None or cost < min_cost:
                min_cost = cost
                min_latency = new_latency
                min_ckpt_time = new_ckpt_time
                min_nodes = selected_nodes

        cache[key] = (min_nodes, min_latency, min_ckpt_time, min_cost)
        return min_nodes, min_latency, min_ckpt_time, min_cost


def dfs(node: Node, visited: set[Node]):
    if node in visited:
        return

    visited.add(node)
    if len(node.parents) > 1:
        checkpoint_name = str(uuid.uuid4())
        for parent in node.parents:
            parent.checkpoint_sets[node.id] = checkpoint_name
            dynamodb[checkpoint_name] = {"values": set(), "size": len(node.parents)}
    for child in node.children:
        dfs(child, visited)
    for parent in node.parents:
        dfs(parent, visited)


def connect(node1: Node, node2: Node):
    e = Edge(node1, node2)
    node1.out_degrees.append(e)
    node2.in_degrees.append(e)


def main():

    app = Graph("example-graph.json")
    print(app.stats())
    app.partition_by_dp(0.5)
    print(app.stats())
    # app.partition_by_dp()
    # start = time.time()
    # app.async_start((3, 4))
    # # app.print_nodes()

    # app.wait()
    # print(app.result())
    # print(time.time() - start)
    # print(json.dumps(dynamodb, indent=2))


if __name__ == "__main__":
    main()
