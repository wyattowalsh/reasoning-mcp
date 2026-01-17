"""Reasoning routers for dynamic method selection and compute allocation.

Routers analyze queries and route them to appropriate reasoning methods
based on complexity, task type, and available compute budget.
"""

from reasoning_mcp.routers.auto_think import AUTO_THINK_METADATA, AutoThink
from reasoning_mcp.routers.base import RouterBase, RouterMetadata
from reasoning_mcp.routers.best_route import BEST_ROUTE_METADATA, BestRoute
from reasoning_mcp.routers.graph_router import GRAPH_ROUTER_METADATA, GraphRouter
from reasoning_mcp.routers.mas_router import MAS_ROUTER_METADATA, MasRouter
from reasoning_mcp.routers.rag_router import RAG_ROUTER_METADATA, RagRouter
from reasoning_mcp.routers.router_r1 import ROUTER_R1_METADATA, RouterR1
from reasoning_mcp.routers.self_budgeter import SELF_BUDGETER_METADATA, SelfBudgeter
from reasoning_mcp.routers.think_switcher import THINK_SWITCHER_METADATA, ThinkSwitcher

# All router classes
ROUTERS = {
    "auto_think": AutoThink,
    "self_budgeter": SelfBudgeter,
    "think_switcher": ThinkSwitcher,
    "router_r1": RouterR1,
    "graph_router": GraphRouter,
    "best_route": BestRoute,
    "mas_router": MasRouter,
    "rag_router": RagRouter,
}

# All router metadata
ROUTER_METADATA = {
    "auto_think": AUTO_THINK_METADATA,
    "self_budgeter": SELF_BUDGETER_METADATA,
    "think_switcher": THINK_SWITCHER_METADATA,
    "router_r1": ROUTER_R1_METADATA,
    "graph_router": GRAPH_ROUTER_METADATA,
    "best_route": BEST_ROUTE_METADATA,
    "mas_router": MAS_ROUTER_METADATA,
    "rag_router": RAG_ROUTER_METADATA,
}

__all__ = [
    # Base
    "RouterBase",
    "RouterMetadata",
    # Routers
    "AutoThink",
    "SelfBudgeter",
    "ThinkSwitcher",
    "RouterR1",
    "GraphRouter",
    "BestRoute",
    "MasRouter",
    "RagRouter",
    # Metadata
    "AUTO_THINK_METADATA",
    "SELF_BUDGETER_METADATA",
    "THINK_SWITCHER_METADATA",
    "ROUTER_R1_METADATA",
    "GRAPH_ROUTER_METADATA",
    "BEST_ROUTE_METADATA",
    "MAS_ROUTER_METADATA",
    "RAG_ROUTER_METADATA",
    # Registries
    "ROUTERS",
    "ROUTER_METADATA",
]
