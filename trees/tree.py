from treelib import Tree
from typing import Optional


def print_tree(tree: Tree, data_property: Optional[str]=None):
    tree_str = tree.show(stdout=False, data_property=data_property)
    print(tree_str)

def example_tree():
    tree = Tree()
    tree.create_node("Node Root", "")  # root node
    tree.create_node("Node 0",    "0",    parent="")
    tree.create_node("Node 1",    "1",    parent="")
    tree.create_node("Node 2", "2", parent="")
    tree.create_node("Node 00",   "00",   parent="0")
    tree.create_node("Node 01",   "01",   parent="0")
    tree.create_node("Node 10",   "10",   parent="1")
    tree.create_node("Node 100",  "100",  parent="10")
    tree.create_node("Node 101",  "101",  parent="10")
    tree.create_node("Node 1010", "1010", parent="101")
    tree.create_node("Node 1011", "1011", parent="101")
    tree.create_node("Node 1012", "1012", parent="101")
    return tree
