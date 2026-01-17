"""Thought graph data models.

This module defines the core data structures for representing thoughts and their
relationships in a reasoning process. ThoughtNode represents individual thoughts
with their metadata, while ThoughtEdge represents connections between thoughts.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# Default maximum sizes for graph collections to prevent memory leaks
DEFAULT_MAX_NODES = 10000
DEFAULT_MAX_EDGES = 20000

from reasoning_mcp.models.core import MethodIdentifier, ThoughtType

if TYPE_CHECKING:
    from reasoning_mcp.utils.graph_utils import ThoughtGraphNetworkX


class ThoughtNode(BaseModel):
    """An immutable node representing a single thought in a reasoning process.

    ThoughtNode captures all information about a single thought, including its
    content, relationships to other thoughts, evaluation metrics, and metadata.
    Nodes are immutable (frozen) to ensure thread-safety and enable safe sharing
    across concurrent reasoning processes.

    Examples:
        Create a new thought:
        >>> thought = ThoughtNode(
        ...     id=str(uuid4()),
        ...     type=ThoughtType.INITIAL,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="Let's break down this problem step by step.",
        ...     step_number=1
        ... )

        Add a child thought:
        >>> child_id = str(uuid4())
        >>> updated_thought = thought.with_child(child_id)

        Update thought properties:
        >>> refined_thought = thought.with_update(
        ...     confidence=0.95,
        ...     quality_score=0.9
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # Identity fields
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this thought (UUID)",
    )
    type: ThoughtType = Field(
        description="Type of thought (e.g., initial, continuation, branch)",
    )
    method_id: MethodIdentifier = Field(
        description="Reasoning method that created this thought",
    )

    # Content fields
    content: str = Field(
        description="The actual thought content - main text of this thought",
    )
    summary: str | None = Field(
        default=None,
        description="Optional brief summary of the thought content",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence, facts, or observations for this thought",
    )

    # Relationship fields
    parent_id: str | None = Field(
        default=None,
        description="ID of the parent thought, if this is a child thought",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child thoughts that derive from this one",
    )
    branch_id: str | None = Field(
        default=None,
        description="Branch identifier if this thought is part of a specific reasoning branch",
    )

    # Evaluation fields
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this thought (0.0 = no confidence, 1.0 = complete confidence)",
    )
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional quality assessment score (0.0 = lowest quality, 1.0 = highest quality)",
    )
    is_valid: bool = Field(
        default=True,
        description="Whether this thought is considered valid in the current reasoning state",
    )

    # Metadata fields
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when this thought was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for method-specific information",
    )
    step_number: int = Field(
        default=0,
        ge=0,
        description="Sequential step number in the reasoning process",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Depth of this thought in the reasoning tree (0 = root)",
    )

    def with_child(self, child_id: str) -> ThoughtNode:
        """Create a new ThoughtNode with an additional child ID.

        Since ThoughtNode is immutable (frozen), this method returns a new
        instance with the child_id added to the children_ids list.

        Args:
            child_id: The ID of the child thought to add

        Returns:
            A new ThoughtNode instance with the child ID added

        Examples:
            >>> child_id = str(uuid4())
            >>> updated = thought.with_child(child_id)
            >>> assert child_id in updated.children_ids
            >>> assert thought.children_ids != updated.children_ids  # Original unchanged
        """
        return self.model_copy(
            update={"children_ids": [*self.children_ids, child_id]},
            deep=True,
        )

    def with_update(self, **kwargs: Any) -> ThoughtNode:
        """Create a new ThoughtNode with updated field values.

        Since ThoughtNode is immutable (frozen), this method returns a new
        instance with the specified fields updated.

        Args:
            **kwargs: Field names and their new values to update

        Returns:
            A new ThoughtNode instance with updated values

        Raises:
            ValueError: If attempting to update non-existent fields

        Examples:
            >>> updated = thought.with_update(
            ...     confidence=0.95,
            ...     quality_score=0.9,
            ...     is_valid=True
            ... )
            >>> assert updated.confidence == 0.95
            >>> assert thought.confidence != updated.confidence  # Original unchanged
        """
        return self.model_copy(update=kwargs, deep=True)


class ThoughtEdge(BaseModel):
    """An immutable edge representing a relationship between two thoughts.

    ThoughtEdge captures directed relationships between thoughts in a reasoning
    graph, including the type of relationship and its strength. Edges are
    immutable (frozen) to ensure consistency in the reasoning graph structure.

    Common edge types include:
        - "derives": Target derives from source
        - "supports": Source provides supporting evidence for target
        - "contradicts": Source contradicts target
        - "branches": Target is a branch from source
        - "refines": Target refines or improves source
        - "questions": Target questions or challenges source

    Examples:
        Create a derivation edge:
        >>> edge = ThoughtEdge(
        ...     id=str(uuid4()),
        ...     source_id=parent_thought.id,
        ...     target_id=child_thought.id,
        ...     edge_type="derives",
        ...     weight=1.0
        ... )

        Create a supporting edge with custom metadata:
        >>> support_edge = ThoughtEdge(
        ...     id=str(uuid4()),
        ...     source_id=evidence_thought.id,
        ...     target_id=conclusion_thought.id,
        ...     edge_type="supports",
        ...     weight=0.8,
        ...     metadata={"strength": "strong", "domain": "empirical"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this edge (UUID)",
    )
    source_id: str = Field(
        description="ID of the source thought (where the edge originates)",
    )
    target_id: str = Field(
        description="ID of the target thought (where the edge points to)",
    )
    edge_type: str = Field(
        description="Type of relationship (e.g., 'derives', 'supports', 'contradicts', 'branches')",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight or strength of this relationship (0.0 = weakest, higher = stronger)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for additional edge information",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when this edge was created",
    )


class ThoughtGraph(BaseModel):
    """A mutable graph representing a collection of thoughts and their relationships.

    ThoughtGraph maintains a directed graph structure of ThoughtNodes connected by
    ThoughtEdges. It provides methods for traversing the graph, finding paths,
    analyzing branches, and computing graph properties.

    Unlike ThoughtNode and ThoughtEdge which are immutable, ThoughtGraph is mutable
    to allow efficient graph construction and modification during reasoning processes.

    The graph has configurable memory bounds (max_nodes, max_edges) to prevent
    unbounded memory growth. When limits are reached, oldest nodes/edges are
    evicted using FIFO order.

    Examples:
        Create and populate a graph:
        >>> graph = ThoughtGraph()
        >>> root = ThoughtNode(
        ...     id="root",
        ...     type=ThoughtType.INITIAL,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="Starting thought"
        ... )
        >>> graph.add_thought(root)
        >>> child = ThoughtNode(
        ...     id="child1",
        ...     type=ThoughtType.CONTINUATION,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="Next thought",
        ...     parent_id="root",
        ...     depth=1
        ... )
        >>> graph.add_thought(child)

        Traverse the graph:
        >>> path = graph.get_path("root", "child1")
        >>> ancestors = graph.get_ancestors("child1")
        >>> descendants = graph.get_descendants("root")

        Analyze graph structure:
        >>> print(f"Graph has {graph.node_count} nodes and {graph.edge_count} edges")
        >>> print(f"Maximum depth: {graph.max_depth}")
        >>> print(f"Number of branches: {graph.branch_count}")

        Create a graph with custom limits:
        >>> small_graph = ThoughtGraph(max_nodes=100, max_edges=200)
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this graph (UUID)",
    )
    nodes: dict[str, ThoughtNode] = Field(
        default_factory=dict,
        description="Map of node ID to ThoughtNode",
    )
    edges: dict[str, ThoughtEdge] = Field(
        default_factory=dict,
        description="Map of edge ID to ThoughtEdge",
    )
    root_id: str | None = Field(
        default=None,
        description="ID of the root thought (entry point of the reasoning process)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for graph-level information",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when this graph was created",
    )
    max_nodes: int = Field(
        default=DEFAULT_MAX_NODES,
        ge=1,
        description="Maximum number of nodes to retain (oldest evicted first)",
    )
    max_edges: int = Field(
        default=DEFAULT_MAX_EDGES,
        ge=1,
        description="Maximum number of edges to retain (oldest evicted first)",
    )

    # Private attributes for tracking insertion order (for FIFO eviction)
    _node_order: deque[str] = PrivateAttr(default_factory=deque)
    _edge_order: deque[str] = PrivateAttr(default_factory=deque)

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph.

        Returns:
            The total count of ThoughtNodes in the graph

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="test"))
            >>> assert graph.node_count == 1
        """
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph.

        Returns:
            The total count of ThoughtEdges in the graph

        Examples:
            >>> graph = ThoughtGraph()
            >>> edge = ThoughtEdge(id="e1", source_id="1", target_id="2", edge_type="derives")
            >>> graph.add_edge(edge)
            >>> assert graph.edge_count == 1
        """
        return len(self.edges)

    @property
    def max_depth(self) -> int:
        """Return the maximum depth across all nodes in the graph.

        Returns:
            The highest depth value among all nodes, or 0 if the graph is empty

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root", depth=0))
            >>> graph.add_thought(ThoughtNode(id="2", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child", depth=2))
            >>> assert graph.max_depth == 2
        """
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())

    @property
    def branch_count(self) -> int:
        """Return the count of unique branches in the graph.

        Counts the number of distinct non-None branch_id values across all nodes.

        Returns:
            The number of unique branches

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root"))
            >>> graph.add_thought(ThoughtNode(id="2", type=ThoughtType.BRANCH, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="branch1", branch_id="b1"))
            >>> graph.add_thought(ThoughtNode(id="3", type=ThoughtType.BRANCH, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="branch2", branch_id="b2"))
            >>> assert graph.branch_count == 2
        """
        branch_ids = {node.branch_id for node in self.nodes.values() if node.branch_id is not None}
        return len(branch_ids)

    @property
    def leaf_ids(self) -> list[str]:
        """Return a list of node IDs that are leaves (have no children).

        Leaf nodes are nodes that have no outgoing edges (empty children_ids list).

        Returns:
            List of node IDs representing leaf nodes

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root"))
            >>> graph.add_thought(ThoughtNode(id="2", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child", parent_id="1"))
            >>> leaves = graph.leaf_ids
            >>> assert "2" in leaves and "1" not in leaves
        """
        return [node_id for node_id, node in self.nodes.items() if not node.children_ids]

    def add_thought(self, thought: ThoughtNode) -> ThoughtGraph:
        """Add a ThoughtNode to the graph and update relationships.

        This method adds a node to the graph and automatically:
        - Sets root_id if this is the first node without a parent
        - Updates parent node to include this thought as a child
        - Creates a "derives" edge from parent to child
        - Evicts oldest nodes if max_nodes limit is reached

        Args:
            thought: The ThoughtNode to add to the graph

        Returns:
            Self for method chaining

        Examples:
            >>> graph = ThoughtGraph()
            >>> root = ThoughtNode(id="root", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="Start")
            >>> graph.add_thought(root)
            >>> assert graph.root_id == "root"
            >>>
            >>> child = ThoughtNode(id="child", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="Next", parent_id="root", depth=1)
            >>> graph.add_thought(child)
            >>> assert "child" in graph.nodes["root"].children_ids
        """
        # Evict oldest nodes if at capacity (but never evict the root)
        while len(self.nodes) >= self.max_nodes and self._node_order:
            oldest_id = self._node_order[0]
            # Don't evict the root node
            if oldest_id == self.root_id:
                self._node_order.popleft()
                self._node_order.append(oldest_id)  # Move to end
                if len(self._node_order) <= 1:
                    break  # Only root left, can't evict
                oldest_id = self._node_order[0]
            if oldest_id in self.nodes:
                del self.nodes[oldest_id]
            self._node_order.popleft()

        # Add the node to the graph
        self.nodes[thought.id] = thought
        self._node_order.append(thought.id)

        # Set root if this is the first node without a parent
        if self.root_id is None and thought.parent_id is None:
            self.root_id = thought.id

        # Update parent-child relationships if this thought has a parent
        if thought.parent_id is not None and thought.parent_id in self.nodes:
            parent = self.nodes[thought.parent_id]
            # Update parent to include this child
            updated_parent = parent.with_child(thought.id)
            self.nodes[thought.parent_id] = updated_parent

            # Create a "derives" edge from parent to child
            edge = ThoughtEdge(
                source_id=thought.parent_id,
                target_id=thought.id,
                edge_type="derives",
            )
            self.add_edge(edge)

        return self

    def add_edge(self, edge: ThoughtEdge) -> ThoughtGraph:
        """Add a ThoughtEdge to the graph.

        Evicts oldest edges if max_edges limit is reached using FIFO order.

        Args:
            edge: The ThoughtEdge to add to the graph

        Returns:
            Self for method chaining

        Examples:
            >>> graph = ThoughtGraph()
            >>> edge = ThoughtEdge(id="e1", source_id="1", target_id="2", edge_type="supports")
            >>> graph.add_edge(edge)
            >>> assert "e1" in graph.edges
        """
        # Evict oldest edges if at capacity
        while len(self.edges) >= self.max_edges and self._edge_order:
            oldest_id = self._edge_order.popleft()
            if oldest_id in self.edges:
                del self.edges[oldest_id]

        self.edges[edge.id] = edge
        self._edge_order.append(edge.id)
        return self

    def get_node(self, node_id: str) -> ThoughtNode | None:
        """Retrieve a node by its ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            The ThoughtNode if found, None otherwise

        Examples:
            >>> graph = ThoughtGraph()
            >>> node = ThoughtNode(id="test", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="test")
            >>> graph.add_thought(node)
            >>> retrieved = graph.get_node("test")
            >>> assert retrieved is not None and retrieved.id == "test"
            >>> assert graph.get_node("nonexistent") is None
        """
        return self.nodes.get(node_id)

    def get_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Find a path between two nodes using breadth-first search.

        Uses BFS to find the shortest path from from_id to to_id, following
        the children_ids relationships.

        Args:
            from_id: The starting node ID
            to_id: The target node ID

        Returns:
            List of node IDs representing the path from from_id to to_id (inclusive),
            or None if no path exists or either node doesn't exist

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root"))
            >>> graph.add_thought(ThoughtNode(id="2", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child", parent_id="1", depth=1))
            >>> graph.add_thought(ThoughtNode(id="3", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="grandchild", parent_id="2", depth=2))
            >>> path = graph.get_path("1", "3")
            >>> assert path == ["1", "2", "3"]
            >>> assert graph.get_path("3", "1") is None  # No path in reverse direction
        """
        # Check if both nodes exist
        if from_id not in self.nodes or to_id not in self.nodes:
            return None

        # If source and target are the same
        if from_id == to_id:
            return [from_id]

        # BFS to find path
        queue: deque[tuple[str, list[str]]] = deque([(from_id, [from_id])])
        visited: set[str] = {from_id}

        while queue:
            current_id, path = queue.popleft()
            current_node = self.nodes[current_id]

            # Check all children
            for child_id in current_node.children_ids:
                if child_id == to_id:
                    return path + [child_id]

                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, path + [child_id]))

        return None

    def get_ancestors(self, node_id: str) -> list[str]:
        """Get all ancestor node IDs from immediate parent to root.

        Follows the parent_id chain upwards to collect all ancestors.

        Args:
            node_id: The ID of the node whose ancestors to retrieve

        Returns:
            List of ancestor node IDs in order from immediate parent to root,
            or empty list if node not found or has no ancestors

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="root", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root"))
            >>> graph.add_thought(ThoughtNode(id="child", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child", parent_id="root", depth=1))
            >>> graph.add_thought(ThoughtNode(id="grandchild", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="grandchild", parent_id="child", depth=2))
            >>> ancestors = graph.get_ancestors("grandchild")
            >>> assert ancestors == ["child", "root"]
        """
        if node_id not in self.nodes:
            return []

        ancestors: list[str] = []
        current = self.nodes[node_id]

        while current.parent_id is not None:
            ancestors.append(current.parent_id)
            if current.parent_id not in self.nodes:
                break
            current = self.nodes[current.parent_id]

        return ancestors

    def get_descendants(self, node_id: str) -> list[str]:
        """Get all descendant node IDs using breadth-first traversal.

        Traverses the graph following children_ids relationships to collect
        all descendants (children, grandchildren, etc.).

        Args:
            node_id: The ID of the node whose descendants to retrieve

        Returns:
            List of all descendant node IDs,
            or empty list if node not found or has no descendants

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="root", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root"))
            >>> graph.add_thought(ThoughtNode(id="child1", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child1", parent_id="root", depth=1))
            >>> graph.add_thought(ThoughtNode(id="child2", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child2", parent_id="root", depth=1))
            >>> graph.add_thought(ThoughtNode(id="grandchild", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="grandchild", parent_id="child1", depth=2))
            >>> descendants = graph.get_descendants("root")
            >>> assert set(descendants) == {"child1", "child2", "grandchild"}
        """
        if node_id not in self.nodes:
            return []

        descendants: list[str] = []
        queue: deque[str] = deque([node_id])
        visited: set[str] = {node_id}

        while queue:
            current_id = queue.popleft()
            current_node = self.nodes[current_id]

            for child_id in current_node.children_ids:
                if child_id not in visited and child_id in self.nodes:
                    visited.add(child_id)
                    descendants.append(child_id)
                    queue.append(child_id)

        return descendants

    def get_branch(self, branch_id: str) -> list[ThoughtNode]:
        """Get all nodes belonging to a specific branch.

        Args:
            branch_id: The branch identifier to filter by

        Returns:
            List of ThoughtNodes that have the specified branch_id,
            or empty list if no nodes match

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="1", type=ThoughtType.BRANCH, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="thought1", branch_id="branch_a"))
            >>> graph.add_thought(ThoughtNode(id="2", type=ThoughtType.BRANCH, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="thought2", branch_id="branch_a"))
            >>> graph.add_thought(ThoughtNode(id="3", type=ThoughtType.BRANCH, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="thought3", branch_id="branch_b"))
            >>> branch_nodes = graph.get_branch("branch_a")
            >>> assert len(branch_nodes) == 2
            >>> assert all(node.branch_id == "branch_a" for node in branch_nodes)
        """
        return [node for node in self.nodes.values() if node.branch_id == branch_id]

    def get_main_path(self) -> list[str]:
        """Get the path from root to the deepest leaf, following highest confidence.

        Traverses from the root node to a leaf node, choosing the child with the
        highest confidence at each decision point. This represents the "main" or
        most confident reasoning path.

        Returns:
            List of node IDs from root to deepest leaf,
            or empty list if graph is empty or has no root

        Examples:
            >>> graph = ThoughtGraph()
            >>> graph.add_thought(ThoughtNode(id="root", type=ThoughtType.INITIAL, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="root", confidence=1.0))
            >>> graph.add_thought(ThoughtNode(id="child1", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child1", parent_id="root", confidence=0.9, depth=1))
            >>> graph.add_thought(ThoughtNode(id="child2", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="child2", parent_id="root", confidence=0.7, depth=1))
            >>> graph.add_thought(ThoughtNode(id="grandchild", type=ThoughtType.CONTINUATION, method_id=MethodIdentifier.CHAIN_OF_THOUGHT, content="grandchild", parent_id="child1", confidence=0.85, depth=2))
            >>> main_path = graph.get_main_path()
            >>> assert main_path == ["root", "child1", "grandchild"]
        """
        if not self.root_id or self.root_id not in self.nodes:
            return []

        path: list[str] = [self.root_id]
        current_id = self.root_id

        while True:
            current_node = self.nodes[current_id]

            # If no children, we've reached a leaf
            if not current_node.children_ids:
                break

            # Find child with highest confidence
            children = [
                self.nodes[child_id]
                for child_id in current_node.children_ids
                if child_id in self.nodes
            ]
            if not children:
                break

            # Select child with highest confidence (and deepest if tied)
            best_child = max(children, key=lambda n: (n.confidence, n.depth))
            path.append(best_child.id)
            current_id = best_child.id

        return path

    # ==================== NetworkX Integration ====================

    def as_networkx(self) -> ThoughtGraphNetworkX:
        """Get a NetworkX adapter for advanced graph analysis.

        Creates a ThoughtGraphNetworkX adapter that provides NetworkX-powered
        analysis capabilities including centrality measures, path finding,
        cycle detection, and graph metrics.

        Returns:
            ThoughtGraphNetworkX adapter wrapping this graph

        Raises:
            ImportError: If NetworkX is not installed

        Examples:
            >>> adapter = graph.as_networkx()
            >>> if adapter.is_valid_dag():
            ...     order = adapter.topological_order()
            ...     print(f"Reasoning order: {order}")
            >>> metrics = adapter.get_graph_metrics()
            >>> print(f"Graph density: {metrics['density']:.3f}")
        """
        from reasoning_mcp.utils.graph_utils import ThoughtGraphNetworkX

        return ThoughtGraphNetworkX(self)

    def validate_dag(self) -> bool:
        """Validate that this graph is a Directed Acyclic Graph (no cycles).

        A valid reasoning graph should be a DAG where thoughts flow from
        premises to conclusions without circular dependencies.

        Uses NetworkX if available, otherwise falls back to a manual DFS-based
        cycle detection algorithm.

        Returns:
            True if the graph is a valid DAG, False if cycles exist

        Examples:
            >>> if not graph.validate_dag():
            ...     print("Warning: Graph contains cycles!")
        """
        try:
            adapter = self.as_networkx()
            return bool(adapter.is_valid_dag())
        except ImportError:
            # Fallback: Manual DFS-based cycle detection
            return self._manual_dag_check()

    def _manual_dag_check(self) -> bool:
        """Manual DAG validation without NetworkX dependency.

        Uses DFS with three-color marking to detect cycles:
        - WHITE (0): Unvisited
        - GRAY (1): Currently in recursion stack
        - BLACK (2): Fully processed

        Returns:
            True if no cycles found, False otherwise
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors: dict[str, int] = {node_id: WHITE for node_id in self.nodes}

        def dfs(node_id: str) -> bool:
            """Returns True if cycle detected."""
            colors[node_id] = GRAY
            node = self.nodes.get(node_id)
            if node:
                for child_id in node.children_ids:
                    if child_id in colors:
                        if colors[child_id] == GRAY:
                            return True  # Back edge = cycle
                        if colors[child_id] == WHITE and dfs(child_id):
                            return True
            colors[node_id] = BLACK
            return False

        # Check all nodes (handles disconnected components)
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return False  # Cycle found

        return True  # No cycles

    def get_critical_thoughts(self, top_k: int = 5) -> list[str]:
        """Get the most critical thoughts by betweenness centrality.

        Critical thoughts are those that act as "bridges" in the reasoning
        graph, connecting different reasoning chains. High centrality indicates
        thoughts that many reasoning paths pass through.

        Uses NetworkX betweenness centrality if available, otherwise falls back
        to returning the highest confidence thoughts.

        Args:
            top_k: Number of top critical thoughts to return

        Returns:
            List of node IDs representing the most critical thoughts

        Examples:
            >>> critical = graph.get_critical_thoughts(top_k=3)
            >>> for node_id in critical:
            ...     thought = graph.get_node(node_id)
            ...     print(f"Critical: {thought.content[:50]}...")
        """
        try:
            adapter = self.as_networkx()
            return [node_id for node_id, _ in adapter.get_critical_nodes(top_k)]
        except ImportError:
            # Fallback: Return highest confidence thoughts
            sorted_nodes = sorted(
                self.nodes.values(),
                key=lambda n: (n.confidence, -n.depth),
                reverse=True,
            )
            return [n.id for n in sorted_nodes[:top_k]]

    def find_reasoning_cycles(self) -> list[list[str]]:
        """Find all cycles in the reasoning graph.

        Cycles in a reasoning graph may indicate circular logic or
        logical dependencies that need resolution.

        Uses NetworkX if available for efficient cycle detection.

        Returns:
            List of cycles, where each cycle is a list of node IDs

        Examples:
            >>> cycles = graph.find_reasoning_cycles()
            >>> if cycles:
            ...     print(f"Found {len(cycles)} logical cycles")
            ...     for cycle in cycles:
            ...         print(f"  Cycle: {' -> '.join(cycle)}")
        """
        try:
            adapter = self.as_networkx()
            cycles: list[list[str]] = adapter.find_cycles()
            return cycles
        except ImportError:
            # Fallback: Return empty list (would need manual cycle enumeration)
            return []

    def get_reasoning_clusters(self) -> list[set[str]]:
        """Find independent reasoning clusters (weakly connected components).

        Reasoning clusters are groups of thoughts that are connected to each
        other but disconnected from other clusters. Multiple clusters may
        indicate parallel reasoning threads.

        Uses NetworkX if available.

        Returns:
            List of sets, where each set contains node IDs in a cluster

        Examples:
            >>> clusters = graph.get_reasoning_clusters()
            >>> print(f"Found {len(clusters)} independent reasoning chains")
        """
        try:
            adapter = self.as_networkx()
            clusters: list[set[str]] = adapter.get_reasoning_clusters()
            return clusters
        except ImportError:
            # Fallback: Return single cluster with all nodes
            if self.nodes:
                return [set(self.nodes.keys())]
            return []

    def get_graph_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the reasoning graph.

        Returns a dictionary with various graph statistics useful for
        understanding the structure and complexity of the reasoning process.

        Uses NetworkX if available for advanced metrics, otherwise returns
        basic metrics computed manually.

        Returns:
            Dictionary containing graph metrics:
            - nodes: Number of nodes
            - edges: Number of edges
            - max_depth: Maximum depth in the graph
            - branch_count: Number of unique branches
            - is_dag: Whether the graph is acyclic (if NetworkX available)
            - density: Graph density (if NetworkX available)
            - connected_components: Number of components (if NetworkX available)

        Examples:
            >>> metrics = graph.get_graph_metrics()
            >>> print(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
            >>> print(f"Is DAG: {metrics.get('is_dag', 'unknown')}")
        """
        # Basic metrics always available
        metrics: dict[str, Any] = {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "max_depth": self.max_depth,
            "branch_count": self.branch_count,
            "leaf_count": len(self.leaf_ids),
        }

        # Add NetworkX metrics if available
        try:
            adapter = self.as_networkx()
            nx_metrics = adapter.get_graph_metrics()
            metrics.update(nx_metrics)
        except ImportError:
            # Add basic DAG check without NetworkX
            metrics["is_dag"] = self.validate_dag()

        return metrics
