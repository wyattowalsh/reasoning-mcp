"""CLI inspect command for examining reasoning-mcp components.

This module provides inspection capabilities for methods, sessions, and plugins,
with support for rich terminal formatting and JSON output for scripting.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext

from rich.table import Table

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory
from reasoning_mcp.models.session import Session

console = Console()


def format_method_table(methods: list[MethodMetadata]) -> None:
    """Format a list of methods as a table for terminal display.

    Creates a formatted table with method details including identifier, name,
    category, complexity, and feature support indicators.

    Args:
        methods: List of MethodMetadata objects to format
    """
    table = Table(title="Registered Reasoning Methods", show_header=True, header_style="bold")
    table.add_column("Identifier", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Complexity", justify="center")
    table.add_column("Branching", justify="center")
    table.add_column("Revision", justify="center")

    for method in methods:
        # Use checkmarks for boolean flags
        branching = "✓" if method.supports_branching else "✗"
        revision = "✓" if method.supports_revision else "✗"

        table.add_row(
            str(method.identifier),
            method.name,
            str(method.category),
            str(method.complexity),
            branching,
            revision,
        )

    console.print(table)


def format_session_info(session: Session) -> None:
    """Format session information for terminal output.

    Creates a formatted display of session details including status, metrics,
    and thought graph statistics.

    Args:
        session: Session object to format
    """
    table = Table(title=f"Session: {session.id}", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Status", str(session.status))
    table.add_row("Created", session.created_at.isoformat())

    # Add started/completed times if available
    if session.started_at:
        table.add_row("Started", session.started_at.isoformat())
    if session.completed_at:
        table.add_row("Completed", session.completed_at.isoformat())

    # Duration
    if session.duration is not None:
        table.add_row("Duration", f"{session.duration:.2f}s")

    # Metrics
    table.add_row("Total Thoughts", str(session.metrics.total_thoughts))
    table.add_row("Total Edges", str(session.metrics.total_edges))
    table.add_row("Max Depth", str(session.metrics.max_depth_reached))
    table.add_row("Avg Confidence", f"{session.metrics.average_confidence:.2f}")

    # Quality if available
    if session.metrics.average_quality is not None:
        table.add_row("Avg Quality", f"{session.metrics.average_quality:.2f}")

    # Branch metrics
    table.add_row("Branches Created", str(session.metrics.branches_created))
    table.add_row("Branches Merged", str(session.metrics.branches_merged))
    table.add_row("Branches Pruned", str(session.metrics.branches_pruned))

    # Error if present
    if session.error:
        table.add_row("Error", session.error)

    console.print(table)


def format_plugin_info(plugin: dict[str, Any]) -> None:
    """Format plugin information for terminal output.

    Creates a formatted display of plugin details including name, version,
    status, and provided methods.

    Args:
        plugin: Dictionary containing plugin information
    """
    table = Table(
        title=f"Plugin: {plugin.get('name', 'Unknown')}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Name", plugin.get("name", "Unknown"))
    table.add_row("Version", plugin.get("version", "Unknown"))

    # Status with color coding
    status = plugin.get("status", "unknown")
    status_color = "green" if status == "healthy" else "red" if status == "unhealthy" else "yellow"
    table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")

    # Methods provided
    methods = plugin.get("methods", [])
    table.add_row("Methods Provided", str(len(methods)))
    if methods:
        table.add_row("Method List", ", ".join(methods))

    # Additional info if available
    if "description" in plugin:
        table.add_row("Description", plugin["description"])

    console.print(table)


def inspect_target(
    ctx: CLIContext,
    target: str | None,
    inspect_type: str | None,
) -> None:
    """Inspect sessions, methods, or pipelines.

    Args:
        ctx: CLI context containing settings, logger, and registry.
        target: Optional target to inspect (ID, name, etc.).
        inspect_type: Type of object to inspect (session, method, pipeline).
    """
    # Determine output format from context
    json_output = ctx.verbose >= 2  # Use JSON for very verbose output

    # If no type specified, show help
    if inspect_type is None:
        console.print("Please specify inspection type with --type:")
        console.print("  --type method   : Inspect reasoning methods")
        console.print("  --type session  : Inspect reasoning sessions")
        console.print("  --type pipeline : Inspect reasoning pipelines")
        return

    # Route to appropriate inspection handler
    if inspect_type.lower() == "method":
        _inspect_methods(ctx, target, json_output)
    elif inspect_type.lower() == "session":
        _inspect_sessions(ctx, target, json_output)
    elif inspect_type.lower() == "pipeline":
        _inspect_pipelines(ctx, target, json_output)
    else:
        ctx.logger.error(f"Unknown inspection type: {inspect_type}")
        console.print(f"[red]Invalid inspection type: {inspect_type}[/red]")
        raise typer.Exit(1)


def _inspect_methods(ctx: CLIContext, target: str | None, json_output: bool) -> None:
    """Inspect reasoning methods.

    Args:
        ctx: CLI context
        target: Optional method identifier to inspect
        json_output: Whether to output as JSON
    """
    from reasoning_mcp.models.core import MethodIdentifier

    # Get methods from registry (placeholder data for now)
    example_methods = [
        MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step reasoning with intermediate steps shown",
            category=MethodCategory.CORE,
            complexity=3,
            supports_branching=False,
            supports_revision=True,
        ),
        MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="Tree of Thoughts",
            description="Explore multiple reasoning paths in a tree structure",
            category=MethodCategory.CORE,
            complexity=7,
            supports_branching=True,
            supports_revision=True,
        ),
    ]

    # Filter by target if specified
    if target:
        filtered = [m for m in example_methods if str(m.identifier) == target or m.name == target]
        if not filtered:
            ctx.logger.error(f"Method not found: {target}")
            console.print(f"[red]Method not found: {target}[/red]")
            raise typer.Exit(1)
        methods = filtered
    else:
        methods = example_methods

    # Output
    if json_output:
        methods_data = []
        for method in methods:
            method_dict = {
                "identifier": str(method.identifier),
                "name": method.name,
                "description": method.description,
                "category": str(method.category),
                "complexity": method.complexity,
                "supports_branching": method.supports_branching,
                "supports_revision": method.supports_revision,
                "requires_context": method.requires_context,
                "min_thoughts": method.min_thoughts,
                "max_thoughts": method.max_thoughts,
                "avg_tokens_per_thought": method.avg_tokens_per_thought,
                "tags": list(method.tags),
                "best_for": list(method.best_for),
                "not_recommended_for": list(method.not_recommended_for),
            }
            methods_data.append(method_dict)
        console.print(json.dumps(methods_data, indent=2))
    else:
        format_method_table(methods)

        # Show detailed info in verbose mode
        if ctx.verbose > 0:
            for method in methods:
                console.print(f"\n[bold]{method.name}:[/bold]")
                console.print(f"  Description: {method.description}")
                console.print(f"  Tags: {', '.join(method.tags) if method.tags else 'None'}")
                console.print(f"  Complexity: {method.complexity}/10")
                console.print(f"  Context Required: {method.requires_context}")
                console.print(f"  Thought Range: {method.min_thoughts}-{method.max_thoughts}")
                if method.best_for:
                    console.print(f"  Best For: {', '.join(method.best_for)}")


def _inspect_sessions(ctx: CLIContext, target: str | None, json_output: bool) -> None:
    """Inspect reasoning sessions.

    Args:
        ctx: CLI context
        target: Optional session ID to inspect
        json_output: Whether to output as JSON
    """
    if target:
        # Show specific session
        session = Session()  # Placeholder
        session.start()

        if json_output:
            session_data = {
                "id": session.id,
                "status": str(session.status),
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "duration": session.duration,
                "thought_count": session.thought_count,
                "metrics": {
                    "total_thoughts": session.metrics.total_thoughts,
                    "total_edges": session.metrics.total_edges,
                    "max_depth": session.metrics.max_depth_reached,
                    "average_confidence": session.metrics.average_confidence,
                    "average_quality": session.metrics.average_quality,
                    "branches_created": session.metrics.branches_created,
                    "branches_merged": session.metrics.branches_merged,
                    "branches_pruned": session.metrics.branches_pruned,
                },
            }
            console.print(json.dumps(session_data, indent=2))
        else:
            format_session_info(session)
    else:
        # List all sessions
        ctx.logger.warning("No active sessions found")
        console.print("No active sessions found.")
        console.print("\nTip: Create a session using the appropriate MCP tools")


def _inspect_pipelines(ctx: CLIContext, target: str | None, json_output: bool) -> None:
    """Inspect reasoning pipelines.

    Args:
        ctx: CLI context
        target: Optional pipeline ID to inspect
        json_output: Whether to output as JSON
    """
    ctx.logger.warning("Pipeline inspection not yet implemented")
    console.print("Pipeline inspection is not yet implemented.")
    console.print("\nThis feature will be available in a future release.")


__all__ = [
    "inspect_target",
    "format_method_table",
    "format_session_info",
    "format_plugin_info",
]
