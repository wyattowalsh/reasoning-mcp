"""NetworkX adapter for ThoughtGraph analysis.

This module provides a NetworkX-backed adapter for advanced graph analysis
of ThoughtGraphs. It enables powerful graph algorithms like cycle detection,
centrality analysis, path finding, and topological sorting.

Optional Dependency:
    NetworkX must be installed for this module to work:
    >>> pip install reasoning-mcp[graphs]

Example:
    >>> from reasoning_mcp.utils.graph_utils import ThoughtGraphNetworkX
    >>> adapter = ThoughtGraphNetworkX(thought_graph)
    >>> is_dag = adapter.is_valid_dag()
    >>> critical = adapter.get_critical_nodes(top_k=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from reasoning_mcp.models.thought import ThoughtGraph

logger = logging.getLogger(__name__)

# Try to import NetworkX
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class ThoughtGraphNetworkX:
    """NetworkX-backed analysis for ThoughtGraph.

    ThoughtGraphNetworkX provides advanced graph analysis capabilities by
    converting a ThoughtGraph to a NetworkX DiGraph. Features include:

    - DAG validation (cycle detection)
    - Centrality analysis (betweenness, pagerank)
    - Path finding (all paths between nodes)
    - Topological sorting
    - Connected component analysis
    - Graph metrics (density, clustering)
    - Export to various formats (GraphML, GEXF)

    The adapter lazily builds the NetworkX graph on first access and caches
    it for subsequent operations. Call invalidate_cache() after modifying
    the underlying ThoughtGraph.

    Examples:
        Basic usage:
        >>> adapter = ThoughtGraphNetworkX(thought_graph)
        >>> if adapter.is_valid_dag():
        ...     order = adapter.topological_order()
        ...     print(f"Reasoning order: {order}")

        Find critical thoughts:
        >>> critical = adapter.get_critical_nodes(top_k=5)
        >>> for node_id, score in critical:
        ...     print(f"{node_id}: {score:.3f}")

        Analyze paths:
        >>> paths = adapter.get_all_paths("root", "conclusion")
        >>> print(f"Found {len(paths)} reasoning paths")
    """

    def __init__(self, thought_graph: ThoughtGraph):
        """Initialize the NetworkX adapter.

        Args:
            thought_graph: The ThoughtGraph to analyze

        Raises:
            ImportError: If NetworkX is not installed
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for graph analysis. "
                "Install with: pip install reasoning-mcp[graphs]"
            )

        self.thought_graph = thought_graph
        self._nx_graph: nx.DiGraph | None = None

    def invalidate_cache(self) -> None:
        """Invalidate the cached NetworkX graph.

        Call this after modifying the underlying ThoughtGraph to ensure
        subsequent analysis uses the updated graph structure.
        """
        self._nx_graph = None

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Get or build the NetworkX DiGraph.

        Lazily builds the NetworkX graph from the ThoughtGraph on first access.
        The graph is cached for subsequent operations.

        Returns:
            NetworkX DiGraph representation of the thought graph
        """
        if self._nx_graph is None:
            self._nx_graph = self._build_nx_graph()
        return self._nx_graph

    def _build_nx_graph(self) -> nx.DiGraph:
        """Build a NetworkX DiGraph from the ThoughtGraph.

        Creates nodes with attributes from ThoughtNode fields and edges
        with attributes from ThoughtEdge fields.

        Returns:
            Populated NetworkX DiGraph
        """
        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.thought_graph.nodes.items():
            G.add_node(
                node_id,
                content=node.content[:100] if node.content else "",  # Truncate for memory
                type=node.type.value if hasattr(node.type, "value") else str(node.type),
                method_id=str(node.method_id),
                confidence=node.confidence,
                quality=node.quality_score or 0.0,
                depth=node.depth,
                is_valid=node.is_valid,
                branch_id=node.branch_id,
            )

        # Add edges with attributes
        for edge_id, edge in self.thought_graph.edges.items():
            G.add_edge(
                edge.source_id,
                edge.target_id,
                id=edge_id,
                type=edge.edge_type,
                weight=edge.weight,
            )

        return G

    # ==================== DAG Validation ====================

    def is_valid_dag(self) -> bool:
        """Check if the graph is a valid Directed Acyclic Graph.

        A valid DAG has no cycles, which is required for proper reasoning
        flow where conclusions should not depend on themselves.

        Returns:
            True if the graph is a DAG, False if cycles exist

        Examples:
            >>> if not adapter.is_valid_dag():
            ...     cycles = adapter.find_cycles()
            ...     print(f"Warning: Found {len(cycles)} cycles")
        """
        return cast("bool", nx.is_directed_acyclic_graph(self.nx_graph))

    def find_cycles(self) -> list[list[str]]:
        """Find all cycles in the graph.

        Identifies circular dependencies in the reasoning graph that may
        indicate logical loops or errors.

        Returns:
            List of cycles, where each cycle is a list of node IDs
            (empty list if graph is a DAG)
        """
        return list(nx.simple_cycles(self.nx_graph))

    # ==================== Centrality Analysis ====================

    def get_critical_nodes(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Find the most critical nodes by betweenness centrality.

        Betweenness centrality measures how often a node lies on shortest
        paths between other nodes. High centrality indicates critical
        "bridge" thoughts that connect different reasoning chains.

        Args:
            top_k: Number of top nodes to return

        Returns:
            List of (node_id, centrality_score) tuples, sorted by score descending

        Examples:
            >>> critical = adapter.get_critical_nodes(top_k=3)
            >>> print("Critical bridge thoughts:")
            >>> for node_id, score in critical:
            ...     print(f"  {node_id}: {score:.3f}")
        """
        centrality = nx.betweenness_centrality(self.nx_graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]

    def get_pagerank(self, top_k: int = 5, alpha: float = 0.85) -> list[tuple[str, float]]:
        """Get PageRank scores for nodes.

        PageRank measures the "importance" of nodes based on incoming connections.
        Useful for finding thoughts that are referenced/built upon most.

        Args:
            top_k: Number of top nodes to return
            alpha: Damping factor (default: 0.85)

        Returns:
            List of (node_id, pagerank_score) tuples, sorted by score descending
        """
        try:
            pagerank = nx.pagerank(self.nx_graph, alpha=alpha)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_k]
        except nx.PowerIterationFailedConvergence:
            # Fall back to in-degree for non-converging graphs
            logger.warning("PageRank failed to converge, using in-degree instead")
            in_degrees = dict(self.nx_graph.in_degree())
            sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_k]

    # ==================== Path Analysis ====================

    def get_all_paths(
        self,
        source: str,
        target: str,
        max_length: int | None = None,
    ) -> list[list[str]]:
        """Find all simple paths between two nodes.

        Useful for understanding all possible reasoning chains from one
        thought to another.

        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length (None for unlimited)

        Returns:
            List of paths, where each path is a list of node IDs
        """
        if source not in self.nx_graph or target not in self.nx_graph:
            return []

        cutoff = max_length if max_length else len(self.nx_graph)
        return list(nx.all_simple_paths(self.nx_graph, source, target, cutoff=cutoff))

    def get_shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find the shortest path between two nodes.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Shortest path as list of node IDs, or None if no path exists
        """
        try:
            return cast("list[str]", nx.shortest_path(self.nx_graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_shortest_path_length(self, source: str, target: str) -> int | None:
        """Get the length of the shortest path between two nodes.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Path length, or None if no path exists
        """
        try:
            return cast("int", nx.shortest_path_length(self.nx_graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ==================== Topological Analysis ====================

    def topological_order(self) -> list[str]:
        """Get nodes in topological order.

        Topological order ensures that each node appears before its
        descendants, representing a valid reasoning order.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If the graph contains cycles
        """
        if not self.is_valid_dag():
            raise ValueError("Cannot compute topological order: graph contains cycles")
        return list(nx.topological_sort(self.nx_graph))

    def topological_generations(self) -> list[list[str]]:
        """Get nodes grouped by topological generation.

        Nodes in the same generation can be processed in parallel as they
        have no dependencies on each other.

        Returns:
            List of generations, where each generation is a list of node IDs

        Raises:
            ValueError: If the graph contains cycles
        """
        if not self.is_valid_dag():
            raise ValueError("Cannot compute generations: graph contains cycles")
        return [list(gen) for gen in nx.topological_generations(self.nx_graph)]

    # ==================== Component Analysis ====================

    def get_reasoning_clusters(self) -> list[set[str]]:
        """Find weakly connected components (reasoning clusters).

        Weakly connected components represent independent reasoning chains
        that don't share any connections.

        Returns:
            List of components, where each component is a set of node IDs
        """
        return list(nx.weakly_connected_components(self.nx_graph))

    def get_strongly_connected_clusters(self) -> list[set[str]]:
        """Find strongly connected components.

        Strongly connected components are groups of nodes where every node
        is reachable from every other node. In a reasoning graph, these
        may indicate circular reasoning patterns.

        Returns:
            List of strongly connected components
        """
        return list(nx.strongly_connected_components(self.nx_graph))

    # ==================== Graph Metrics ====================

    def get_graph_metrics(self) -> dict[str, Any]:
        """Get comprehensive graph metrics.

        Returns:
            Dictionary containing:
            - nodes: Number of nodes
            - edges: Number of edges
            - density: Graph density (0-1)
            - is_dag: Whether graph is a DAG
            - connected_components: Number of weakly connected components
            - avg_clustering: Average clustering coefficient
            - max_depth: Maximum depth in the graph
            - avg_out_degree: Average outgoing edges per node
        """
        G = self.nx_graph
        undirected = G.to_undirected()

        out_degrees = [d for _, d in G.out_degree()]
        avg_out_degree = sum(out_degrees) / len(out_degrees) if out_degrees else 0

        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "connected_components": nx.number_weakly_connected_components(G),
            "avg_clustering": nx.average_clustering(undirected) if G.number_of_nodes() > 0 else 0,
            "max_depth": self.thought_graph.max_depth,
            "avg_out_degree": avg_out_degree,
        }

    def get_diameter(self) -> int | None:
        """Get the diameter of the graph (longest shortest path).

        Returns:
            Diameter, or None if graph is not connected
        """
        try:
            return cast("int", nx.diameter(self.nx_graph.to_undirected()))
        except nx.NetworkXError:
            return None

    # ==================== Node Analysis ====================

    def get_sources(self) -> list[str]:
        """Get source nodes (no incoming edges).

        Source nodes are starting points of reasoning chains.

        Returns:
            List of node IDs with in-degree 0
        """
        return [n for n, d in self.nx_graph.in_degree() if d == 0]

    def get_sinks(self) -> list[str]:
        """Get sink nodes (no outgoing edges).

        Sink nodes are endpoints/conclusions of reasoning chains.

        Returns:
            List of node IDs with out-degree 0
        """
        return [n for n, d in self.nx_graph.out_degree() if d == 0]

    def get_high_confidence_path(
        self,
        source: str | None = None,
        target: str | None = None,
    ) -> list[str]:
        """Find the path with highest cumulative confidence.

        Uses edge weights (based on node confidence) to find the most
        confident reasoning path.

        Args:
            source: Starting node (defaults to root)
            target: Ending node (defaults to highest-confidence sink)

        Returns:
            Path as list of node IDs
        """
        if source is None:
            sources = self.get_sources()
            if not sources:
                return []
            source = sources[0]

        if target is None:
            sinks = self.get_sinks()
            if not sinks:
                return []
            # Pick sink with highest confidence
            sink_confidences = [(s, self.nx_graph.nodes[s].get("confidence", 0)) for s in sinks]
            target = max(sink_confidences, key=lambda x: x[1])[0]

        try:
            # Use Dijkstra with inverted weights (to maximize confidence)
            # First, create a copy with inverted weights
            G_inv = self.nx_graph.copy()
            for u, v, _data in G_inv.edges(data=True):
                target_conf = G_inv.nodes[v].get("confidence", 0)
                # Invert: high confidence = low weight
                G_inv[u][v]["weight"] = 1.0 - target_conf

            return cast("list[str]", nx.dijkstra_path(G_inv, source, target, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ==================== Export ====================

    def export_graphml(self, path: str | Path) -> None:
        """Export the graph to GraphML format.

        GraphML is a standard XML format for graphs that can be opened
        in tools like Gephi, yEd, or Cytoscape for visualization.

        Args:
            path: Output file path
        """
        nx.write_graphml(self.nx_graph, str(path))
        logger.info(f"Exported graph to GraphML: {path}")

    def export_gexf(self, path: str | Path) -> None:
        """Export the graph to GEXF format.

        GEXF is the native format for Gephi visualization software.

        Args:
            path: Output file path
        """
        nx.write_gexf(self.nx_graph, str(path))
        logger.info(f"Exported graph to GEXF: {path}")

    def export_json(self) -> dict[str, Any]:
        """Export the graph to a JSON-compatible dictionary.

        Returns:
            Dictionary with 'nodes' and 'links' keys suitable for D3.js
        """
        return cast("dict[str, Any]", nx.node_link_data(self.nx_graph))


def is_networkx_available() -> bool:
    """Check if NetworkX is available.

    Returns:
        True if NetworkX is installed and can be imported
    """
    return NETWORKX_AVAILABLE
