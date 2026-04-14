"""Ray 相关工具函数"""

import ray
from typing import List, Any


def get_available_nodes() -> List[str]:
    """获取可用的 Ray 节点
    
    Returns:
        可用节点列表
    """
    nodes = ray.nodes()
    available_nodes = [node['NodeID'] for node in nodes if node['Alive']]
    return available_nodes


def get_node_resources(node_id: str) -> dict:
    """获取节点资源
    
    Args:
        node_id: 节点 ID
    
    Returns:
        节点资源字典
    """
    nodes = ray.nodes()
    for node in nodes:
        if node['NodeID'] == node_id:
            return node['Resources']
    return {}


def run_on_node(func: callable, args: tuple, node_id: str) -> Any:
    """在指定节点上运行函数
    
    Args:
        func: 要运行的函数
        args: 函数参数
        node_id: 节点 ID
    
    Returns:
        函数返回值
    """
    @ray.remote(resources={node_id: 1})
    def remote_func(*args):
        return func(*args)
    
    return ray.get(remote_func.remote(*args))
