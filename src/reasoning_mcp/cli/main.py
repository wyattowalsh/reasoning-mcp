"""CLI main module for reasoning-mcp.

This module provides the primary entry point for the reasoning-mcp command-line
interface using Typer. It handles configuration loading, logging setup, and
dispatches to subcommands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from reasoning_mcp.config import Settings, get_settings
from reasoning_mcp.logging import setup_logging
from reasoning_mcp.registry import MethodRegistry

if TYPE_CHECKING:
    import logging

# Create main Typer app
app = typer.Typer(
    name="reasoning-mcp",
    help="reasoning-mcp: Advanced reasoning methods for AI systems.",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global state for CLI context
_cli_context: CLIContext | None = None


class CLIContext:
    """Context object passed to CLI commands.

    Attributes:
        settings: Application settings instance
        logger: Configured logger instance
        registry: Method registry instance
        verbose: Verbosity level (0=normal, 1=verbose, 2=debug)
    """

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        registry: MethodRegistry,
        verbose: int = 0,
    ) -> None:
        """Initialize the CLI context.

        Args:
            settings: Application settings
            logger: Configured logger
            registry: Method registry
            verbose: Verbosity level
        """
        self.settings = settings
        self.logger = logger
        self.registry = registry
        self.verbose = verbose


def get_cli_context() -> CLIContext:
    """Get the current CLI context.

    Returns:
        The active CLI context.

    Raises:
        typer.Exit: If context not initialized.
    """
    global _cli_context
    if _cli_context is None:
        typer.echo("Error: CLI context not initialized", err=True)
        raise typer.Exit(1)
    return _cli_context


def get_settings_from_config(config_path: str | None) -> Settings:
    """Load settings from config file or use defaults.

    Args:
        config_path: Optional path to configuration file. If provided,
            loads settings from the file; otherwise uses defaults and
            environment variables.

    Returns:
        Settings instance with configuration loaded.

    Raises:
        typer.Exit: If config file path is invalid or unreadable.
    """
    if config_path is None:
        return get_settings()

    config_file = Path(config_path)
    if not config_file.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    if not config_file.is_file():
        typer.echo(f"Error: Config path is not a file: {config_path}", err=True)
        raise typer.Exit(1)

    try:
        from reasoning_mcp.config import Settings

        return Settings(_env_file=str(config_file))  # type: ignore[call-arg]
    except Exception as e:
        typer.echo(f"Error: Failed to load config: {e}", err=True)
        raise typer.Exit(1) from e


def setup_logging_from_verbosity(verbosity: int, settings: Settings) -> logging.Logger:
    """Configure logging based on verbosity level.

    Args:
        verbosity: Verbosity count (0=INFO, 1=DEBUG, 2+=DEBUG with more details)
        settings: Settings instance for default log configuration

    Returns:
        Configured logger instance.
    """
    if verbosity == 0:
        log_level = settings.log_level
    elif verbosity == 1:
        log_level = "DEBUG"
    else:
        log_level = "DEBUG"

    return setup_logging(settings=settings, log_level=log_level)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo("reasoning-mcp version 0.1.0")
        raise typer.Exit()


@app.callback()
def main_callback(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (can be used multiple times: -v, -vv)",
        ),
    ] = 0,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """reasoning-mcp: Advanced reasoning methods for AI systems.

    This CLI provides commands to run the MCP server, inspect sessions and methods,
    install plugins, and perform health checks.

    Use --verbose/-v for more detailed output (use -vv for even more detail).
    """
    global _cli_context

    try:
        settings = get_settings_from_config(str(config) if config else None)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1) from e

    logger = setup_logging_from_verbosity(verbose, settings)
    registry = MethodRegistry()

    _cli_context = CLIContext(
        settings=settings,
        logger=logger,
        registry=registry,
        verbose=verbose,
    )

    if verbose > 0:
        logger.debug(f"CLI started with verbosity={verbose}")
        logger.debug(f"Config: {config or 'default'}")


@app.command()
def run(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="Host to bind to",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to",
        ),
    ] = 8000,
    transport: Annotated[
        str,
        typer.Option(
            "--transport",
            "-t",
            help="Transport type: stdio, sse, or websocket",
        ),
    ] = "stdio",
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            "-d",
            help="Enable debug logging",
        ),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            "-r",
            help="Enable auto-reload for development (HTTP/WebSocket only)",
        ),
    ] = False,
) -> None:
    """Start the MCP server.

    Launches the reasoning-mcp server and begins listening for MCP protocol
    messages. The server will run until interrupted (Ctrl+C).

    Examples:
        # Start with stdio transport (default for MCP)
        reasoning-mcp run

        # Start SSE server on custom host/port
        reasoning-mcp run --host 0.0.0.0 --port 9000 --transport sse

        # Enable debug mode
        reasoning-mcp run --debug

        # Development mode with auto-reload
        reasoning-mcp run --transport sse --reload
    """
    from reasoning_mcp.cli.commands.run import run_command

    ctx = get_cli_context()
    run_command(
        ctx=ctx,
        host=host,
        port=port,
        transport=transport,
        debug=debug,
        reload=reload,
    )


@app.command()
def inspect(
    target: Annotated[
        str | None,
        typer.Argument(help="What to inspect (session ID, method name, etc.)"),
    ] = None,
    inspect_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Type of object to inspect",
            case_sensitive=False,
        ),
    ] = None,
) -> None:
    """Inspect sessions, methods, or pipelines.

    TARGET is optional and specifies what to inspect (session ID, method name, etc.).
    If not provided, lists all available objects of the specified type.

    Examples:
        reasoning-mcp inspect --type method
        reasoning-mcp inspect chain_of_thought --type method
        reasoning-mcp inspect session-123 --type session
    """
    from reasoning_mcp.cli.commands.inspect import inspect_target

    ctx = get_cli_context()
    inspect_target(ctx, target, inspect_type)


# Create plugin subcommand group
plugin_app = typer.Typer(help="Manage reasoning-mcp plugins")
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("install")
def plugin_install(
    plugin_name: Annotated[str, typer.Argument(help="Plugin name, path, or GitHub repo")],
    upgrade: Annotated[
        bool,
        typer.Option("--upgrade", "-U", help="Upgrade if already installed"),
    ] = False,
) -> None:
    """Install a plugin from PyPI, GitHub, or local path.

    Supports multiple source formats:
    - PyPI packages: my-plugin
    - Local paths: ./local-plugin
    - GitHub shorthand: github:user/repo or gh:user/repo
    - GitHub user/repo: user/repo
    - GitHub with branch/tag: github:user/repo@v1.0 or gh:user/repo#main
    - Full Git URLs: https://github.com/user/repo.git

    Examples:
        reasoning-mcp plugin install my-plugin
        reasoning-mcp plugin install ./local-plugin
        reasoning-mcp plugin install github:anthropics/reasoning-plugin
        reasoning-mcp plugin install gh:user/repo@v1.0
        reasoning-mcp plugin install user/repo
        reasoning-mcp plugin install my-plugin --upgrade
    """
    from reasoning_mcp.cli.commands.install import install_plugin_impl

    ctx = get_cli_context()
    install_plugin_impl(ctx, plugin_name, upgrade)


@plugin_app.command("list")
def plugin_list() -> None:
    """List installed plugins.

    Examples:
        reasoning-mcp plugin list
    """
    from reasoning_mcp.cli.commands.install import list_plugins_impl

    ctx = get_cli_context()
    list_plugins_impl(ctx)


@plugin_app.command("uninstall")
def plugin_uninstall(
    plugin_name: Annotated[str, typer.Argument(help="Plugin name to uninstall")],
) -> None:
    """Uninstall a plugin.

    Examples:
        reasoning-mcp plugin uninstall my-plugin
    """
    from reasoning_mcp.cli.commands.install import uninstall_plugin_impl

    ctx = get_cli_context()
    uninstall_plugin_impl(ctx, plugin_name)


@app.command()
def health(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed checks"),
    ] = False,
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Attempt to fix issues"),
    ] = False,
) -> None:
    """Check health of reasoning-mcp components.

    Performs comprehensive health checks on all system components including
    core modules, registry, methods, plugins, configuration, and dependencies.

    By default, only shows issues. Use --verbose to see all checks.
    Use --json for machine-readable output.
    Use --fix to attempt automatic repair of fixable issues.

    Examples:
        reasoning-mcp health
        reasoning-mcp health --verbose
        reasoning-mcp health --json
        reasoning-mcp health --fix
    """
    from reasoning_mcp.cli.commands.health import health_check

    ctx = get_cli_context()
    health_check(ctx, as_json=as_json, verbose=verbose, fix=fix)


def main() -> None:
    """Entry point for the CLI application.

    This function is called by the installed console script and handles
    graceful shutdown on interrupts.
    """
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


__all__ = [
    "app",
    "main",
    "CLIContext",
    "get_cli_context",
    "get_settings_from_config",
    "setup_logging_from_verbosity",
]
