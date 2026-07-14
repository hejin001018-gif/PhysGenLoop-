"""问题图结果的分类评分、覆盖率和根失败聚合。

本模块不会覆盖现有 violation 风险分数。图评分回答“prompt 中计划检查的内容完成了
多少、其中多少真正可评估”，现有 Fusion 则回答“最强物理违规风险有多高”。两个
视角同时保留，避免把大量 blocked 节点平均成虚假的物理低分或高分。
"""

from __future__ import annotations

from dataclasses import replace

from .schemas import (
    CategoryEvaluation,
    CriticReport,
    GraphEvaluationSummary,
    NODE_STATUSES,
    NodeResult,
    QUESTION_CATEGORIES,
    QuestionGraph,
)


class QuestionGraphScorer:
    """把逐节点结果聚合为 PQSG 风格 fulfillment 和覆盖感知分数。"""

    def enrich_report(
        self,
        report: CriticReport,
        graph: QuestionGraph,
        results: tuple[NodeResult, ...],
    ) -> CriticReport:
        """返回带图摘要和节点结果的新报告，不原地修改冻结 dataclass。"""

        summary = self.summarize(graph, results)
        return replace(report, graph_evaluation=summary, node_results=results)

    def summarize(
        self,
        graph: QuestionGraph,
        results: tuple[NodeResult, ...],
    ) -> GraphEvaluationSummary:
        """校验结果完整性并计算分类/全局统计。"""

        nodes_by_id = {node.id: node for node in graph.nodes}
        results_by_id: dict[str, NodeResult] = {}
        for result in results:
            if result.node_id not in nodes_by_id:
                raise ValueError(f"result references unknown question node {result.node_id}")
            if result.node_id in results_by_id:
                raise ValueError(f"duplicate node result for {result.node_id}")
            if result.status not in NODE_STATUSES:
                raise ValueError(f"unsupported node status {result.status}")
            results_by_id[result.node_id] = result
        missing = [node.id for node in graph.nodes if node.id not in results_by_id]
        if missing:
            raise ValueError(f"question graph results are incomplete: {missing}")

        categories = {
            category: self._category_evaluation(
                category,
                graph,
                results_by_id,
            )
            for category in QUESTION_CATEGORIES
        }
        total_weight = sum(node.weight for node in graph.nodes)
        fulfilled_weight = sum(
            node.weight * (results_by_id[node.id].direct_score or 0.0)
            for node in graph.nodes
        )
        answered_count = sum(
            result.status in {"yes", "no"} for result in results
        )

        physics = categories["physics"]
        return GraphEvaluationSummary(
            prompt_fulfillment_score=(
                fulfilled_weight / total_weight if total_weight else 0.0
            ),
            physics_plausibility_score=physics.score,
            question_coverage=(answered_count / len(graph.nodes) if graph.nodes else 0.0),
            physics_coverage=physics.coverage,
            categories=categories,
            root_failure_nodes=self._root_failures(graph, results_by_id),
        )

    def _category_evaluation(
        self,
        category: str,
        graph: QuestionGraph,
        results_by_id: dict[str, NodeResult],
    ) -> CategoryEvaluation:
        """计算一个类别的直接分数、全节点 fulfillment 和状态计数。"""

        nodes = [node for node in graph.nodes if node.category == category]
        category_results = [results_by_id[node.id] for node in nodes]
        answered_nodes = [
            node
            for node in nodes
            if results_by_id[node.id].status in {"yes", "no"}
        ]
        answered_weight = sum(node.weight for node in answered_nodes)
        direct_total = sum(
            node.weight * (results_by_id[node.id].direct_score or 0.0)
            for node in answered_nodes
        )
        total_weight = sum(node.weight for node in nodes)
        fulfillment_total = sum(
            node.weight * (results_by_id[node.id].direct_score or 0.0)
            for node in nodes
        )
        counts = {
            status: sum(result.status == status for result in category_results)
            for status in NODE_STATUSES
        }
        return CategoryEvaluation(
            category=category,
            score=(direct_total / answered_weight if answered_weight else None),
            fulfillment_score=(
                fulfillment_total / total_weight if total_weight else 0.0
            ),
            coverage=(len(answered_nodes) / len(nodes) if nodes else 0.0),
            total=len(nodes),
            answered=len(answered_nodes),
            yes=counts["yes"],
            no=counts["no"],
            blocked=counts["blocked"],
            unknown=counts["unknown"],
        )

    def _root_failures(
        self,
        graph: QuestionGraph,
        results_by_id: dict[str, NodeResult],
    ) -> tuple[str, ...]:
        """返回没有失败祖先的直接 No 节点，作为最小可修复根因集合。"""

        nodes_by_id = {node.id: node for node in graph.nodes}
        failed = {
            node_id for node_id, result in results_by_id.items() if result.status == "no"
        }

        def has_failed_ancestor(node_id: str) -> bool:
            pending = list(nodes_by_id[node_id].parent_ids)
            visited: set[str] = set()
            while pending:
                parent_id = pending.pop()
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                if parent_id in failed:
                    return True
                pending.extend(nodes_by_id[parent_id].parent_ids)
            return False

        # 按图中节点顺序输出，便于 Repair Agent 生成稳定、可复现的修复列表。
        return tuple(
            node.id
            for node in graph.nodes
            if node.id in failed and not has_failed_ancestor(node.id)
        )

