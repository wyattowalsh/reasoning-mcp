"""Health check command for reasoning-mcp.

Performs comprehensive health checks on all reasoning-mcp components.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import typer

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext


@dataclass
class HealthCheck:
    """Represents a single health check result.

    Attributes:
        name: Name of the component being checked
        status: Health status (ok, warning, or error)
        message: Human-readable status message
        details: Additional diagnostic information
        fixable: Whether the issue can be automatically fixed
    """

    name: str
    status: Literal["ok", "warning", "error"]
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    fixable: bool = False


async def check_core_imports() -> HealthCheck:
    """Check that core modules can be imported."""
    try:
        from reasoning_mcp import config, logging, registry
        from reasoning_mcp.models import core, pipeline, session, thought
        return HealthCheck(name="Core Imports", status="ok", message="Imported")
    except Exception as e:
        return HealthCheck(name="Core Imports", status="error", message=str(e))


async def check_registry() -> HealthCheck:
    """Check that the method registry is functional."""
    try:
        from reasoning_mcp.registry import MethodRegistry
        r = MethodRegistry()
        return HealthCheck(name="Registry", status="ok", message=f"{r.method_count} methods")
    except Exception as e:
        return HealthCheck(name="Registry", status="error", message=str(e))


async def check_methods() -> list[HealthCheck]:
    """Check health of all registered reasoning methods."""
    try:
        from reasoning_mcp.registry import MethodRegistry
        return [HealthCheck(name=f"Method: {m}", status="ok" if h else "warning", message="OK" if h else "Failed", fixable=not h)
                for m, h in (await MethodRegistry().health_check()).items()]
    except Exception as e:
        return [HealthCheck(name="Methods", status="error", message=str(e))]


async def check_plugins() -> HealthCheck:
    """Check that plugins are loadable."""
    try:
        from reasoning_mcp.config import get_settings
        s = get_settings()
        if not s.enable_plugins:
            return HealthCheck(name="Plugins", status="ok", message="Disabled")
        exists = s.plugins_dir.exists()
        return HealthCheck(name="Plugins", status="ok" if exists else "warning", message="OK" if exists else "Missing", fixable=not exists)
    except Exception as e:
        return HealthCheck(name="Plugins", status="error", message=str(e))


async def check_config() -> HealthCheck:
    """Validate configuration settings."""
    try:
        from reasoning_mcp.config import get_settings
        s = get_settings()
        issues = [i for i, c in [("timeout", s.session_timeout < 60), ("max_sessions", s.max_sessions < 1)] if c]
        return HealthCheck(name="Config", status="warning" if issues else "ok", message=", ".join(issues) or "OK")
    except Exception as e:
        return HealthCheck(name="Config", status="error", message=str(e))


async def check_dependencies() -> HealthCheck:
    """Check that required dependencies are installed."""
    missing = [d for d in ["fastmcp", "pydantic", "pydantic_settings", "structlog", "rich", "click"] if not importlib.util.find_spec(d)]
    return HealthCheck(name="Dependencies", status="error" if missing else "ok", message=f"Missing: {', '.join(missing)}" if missing else "OK",
                       details={"missing": missing} if missing else {}, fixable=bool(missing))


async def fix_issues(checks: list[HealthCheck]) -> dict[str, bool]:
    """Attempt to fix fixable issues."""
    results = {}
    for c in [x for x in checks if x.fixable and x.status != "ok"]:
        try:
            if c.name == "Plugins":
                from reasoning_mcp.config import get_settings
                get_settings().plugins_dir.mkdir(parents=True, exist_ok=True)
                results[c.name] = True
            elif c.name.startswith("Method:"):
                from reasoning_mcp.registry import MethodRegistry
                mid = c.name.split(": ", 1)[1]
                results[c.name] = (await MethodRegistry().initialize(mid)).get(mid, False)
            else:
                results[c.name] = False
        except Exception:
            results[c.name] = False
    return results


def format_output(checks: list[HealthCheck], verbose: bool) -> None:
    """Format and print health check results."""
    from rich.console import Console
    from rich.table import Table

    console, table = Console(), Table(title="Health Check Results")
    for col, style in [("Component", "cyan"), ("Status", "bold"), ("Message", None)]:
        table.add_column(col, style=style)

    icons = {"ok": "[green]✓ OK", "warning": "[yellow]⚠ WARN", "error": "[red]✗ ERROR"}
    for c in (checks if verbose else [x for x in checks if x.status != "ok"]):
        table.add_row(c.name, icons[c.status], c.message)

    console.print(table)
    counts = {s: sum(1 for c in checks if c.status == s) for s in ["ok", "warning", "error"]}
    color = "red" if counts["error"] else "yellow" if counts["warning"] else "green"
    console.print(f"\n[{color}][bold]Summary:[/bold] {counts['ok']} OK, {counts['warning']} warn, {counts['error']} errors[/{color}]")


def format_json_output(checks: list[HealthCheck]) -> None:
    """Format and print health check results as JSON."""
    all_ok, no_errors = all(c.status == "ok" for c in checks), all(c.status != "error" for c in checks)
    print(json.dumps({"status": "healthy" if all_ok else "degraded" if no_errors else "unhealthy",
                      "checks": [{"name": c.name, "status": c.status, "message": c.message, "details": c.details, "fixable": c.fixable} for c in checks],
                      "summary": {s: sum(1 for c in checks if c.status == s) for s in ["ok", "warning", "error"]}}, indent=2))


def health_check(
    ctx: "CLIContext",
    *,
    as_json: bool = False,
    verbose: bool = False,
    fix: bool = False,
) -> None:
    """Check health of reasoning-mcp components.

    Args:
        ctx: CLI context containing settings, logger, and registry.
        as_json: Whether to output as JSON.
        verbose: Whether to show all checks (not just failures).
        fix: Whether to attempt to fix fixable issues.

    Performs comprehensive health checks on all system components:
    - Core modules and imports
    - Method registry
    - Registered reasoning methods
    - Plugin system
    - Configuration
    - Dependencies

    By default, only shows issues. Use verbose=True to see all checks.
    Use as_json=True for machine-readable output.
    Use fix=True to attempt automatic repair of fixable issues.
    """

    async def run_checks() -> list[HealthCheck]:
        """Run all health checks asynchronously."""
        checks = [await check_core_imports(), await check_registry(), await check_config(),
                  await check_dependencies(), await check_plugins()]
        checks.extend(await check_methods())
        return checks

    all_checks = asyncio.run(run_checks())

    if fix:
        fix_results = asyncio.run(fix_issues(all_checks))
        if fix_results and not as_json:
            from rich.console import Console
            console = Console()
            console.print("\n[bold]Fix Results:[/bold]")
            for name, success in fix_results.items():
                console.print(f"  {name}: {'[green]✓ Fixed' if success else '[red]✗ Failed'}")
            all_checks = asyncio.run(run_checks())

    (format_json_output if as_json else lambda c: format_output(c, verbose))(all_checks)

    if any(c.status == "error" for c in all_checks):
        raise typer.Exit(1)
