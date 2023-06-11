from graphviz import Digraph

import kaitorch.activations as A
from kaitorch.core import Scalar


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def plot_model(root, filename=None):
    dot = Digraph(format='png', graph_attr={'rankdir': 'TB'})

    all_nodes = set()
    all_edges = set()

    if not isinstance(root, list):
        root = [root]

    for r in root:
        nodes, edges = trace(r)
        all_nodes = all_nodes.union(nodes)
        all_edges = all_edges.union(edges)
    all_nodes = list(all_nodes)
    all_edges = list(all_edges)

    for n in all_nodes:
        uid = str(id(n))

        dot.node(name=uid,
                 label="{data %.4f | grad %.4f}" % (n.data, n.grad),
                 shape='record')
        if n._op:
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op, uid)

    for n1, n2 in all_edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    if filename:
        dot.render(filename=filename, view=True)
    return dot
