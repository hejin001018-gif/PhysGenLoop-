"""问题图的结构校验与稳定拓扑排序。

本模块只处理图结构，不负责生成问题或读取视频证据。将结构校验独立出来，可以让
模板生成器、未来 VLM Question Generator 和人工标注图共享完全相同的约束，避免
某一种来源绕过 DAG 与边类型检查。
"""

from __future__ import annotations

from collections import defaultdict

from .schemas import QuestionGraph, QuestionNode


class QuestionGraphError(ValueError):
    """问题图包含重复 ID、悬空边、非法依赖或环路时抛出的异常。"""


class QuestionGraphValidator:
    """执行跨节点校验，并按输入顺序稳定地生成拓扑序。"""

    # PQSG 主文允许同类别顺序边，以及 Object→Action、Action→Physics 两种跨层边。
    # Object→Physics 会跳过动作前置条件，Physics→Action 等反向边则破坏层级语义。
    _ALLOWED_CROSS_CATEGORY_EDGES = {
        ("object", "action"),
        ("action", "physics"),
    }

    def validate(self, graph: QuestionGraph) -> None:
        """校验节点主键、父引用、边方向和 DAG 性质。"""

        nodes_by_id: dict[str, QuestionNode] = {}
        for node in graph.nodes:
            if node.id in nodes_by_id:
                raise QuestionGraphError(f"duplicate question node id: {node.id}")
            nodes_by_id[node.id] = node

        for node in graph.nodes:
            if len(set(node.parent_ids)) != len(node.parent_ids):
                raise QuestionGraphError(f"node {node.id} contains duplicate parent ids")
            for parent_id in node.parent_ids:
                if parent_id not in nodes_by_id:
                    raise QuestionGraphError(
                        f"node {node.id} references missing parent {parent_id}"
                    )
                if parent_id == node.id:
                    raise QuestionGraphError(f"node {node.id} cannot depend on itself")
                parent = nodes_by_id[parent_id]
                if (
                    parent.category != node.category
                    and (parent.category, node.category)
                    not in self._ALLOWED_CROSS_CATEGORY_EDGES
                ):
                    raise QuestionGraphError(
                        f"illegal dependency {parent.id}({parent.category}) -> "
                        f"{node.id}({node.category})"
                    )

        # 完整消费拓扑序即可证明不存在环；错误中保留未消费节点，便于定位 VLM 输出。
        ordered = self._topological_ids(graph, nodes_by_id)
        if len(ordered) != len(graph.nodes):
            cyclic = [node.id for node in graph.nodes if node.id not in set(ordered)]
            raise QuestionGraphError(f"question graph contains a cycle: {cyclic}")

    def topological_order(self, graph: QuestionGraph) -> tuple[QuestionNode, ...]:
        """返回稳定拓扑序；同层可选节点保持其在 ``graph.nodes`` 中的原始顺序。"""

        self.validate(graph)
        nodes_by_id = {node.id: node for node in graph.nodes}
        ordered_ids = self._topological_ids(graph, nodes_by_id)
        return tuple(nodes_by_id[node_id] for node_id in ordered_ids)

    def _topological_ids(
        self,
        graph: QuestionGraph,
        nodes_by_id: dict[str, QuestionNode],
    ) -> tuple[str, ...]:
        """使用 Kahn 算法计算 ID 序列，不在此重复生成用户可见错误。"""

        indegree = {node.id: len(node.parent_ids) for node in graph.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        input_order = {node.id: index for index, node in enumerate(graph.nodes)}
        for node in graph.nodes:
            for parent_id in node.parent_ids:
                # validate 调用前可能存在悬空父节点；内部算法仅处理已知节点。
                if parent_id in nodes_by_id:
                    children[parent_id].append(node.id)

        ready = [node.id for node in graph.nodes if indegree[node.id] == 0]
        ordered: list[str] = []
        while ready:
            # 每轮按原始位置选择，确保图来自 VLM 时重复执行仍得到确定顺序。
            ready.sort(key=input_order.__getitem__)
            node_id = ready.pop(0)
            ordered.append(node_id)
            for child_id in children[node_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
        return tuple(ordered)

