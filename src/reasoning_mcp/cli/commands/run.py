"""CLI run command for starting the reasoning-mcp server.

This module provides the 'run' command that starts the MCP server with
configured transport (stdio, HTTP, or WebSocket), loads plugins, and
registers native methods.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext
    from reasoning_mcp.config import Settings

logger = logging.getLogger(__name__)
console = Console()


def run_server(ctx: "CLIContext") -> None:
    """Start the MCP server using the CLI context.

    Args:
        ctx: CLI context containing settings, logger, and registry.
    """
    # Use defaults - the full run command allows overrides
    run(
        host="localhost",
        port=8080,
        transport="stdio",
        plugin_dir=[],
        no_plugins=not ctx.settings.enable_plugins,
        debug=ctx.settings.debug,
        log_level=ctx.settings.log_level,
    )


def run(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type: stdio, http, or websocket",
    ),
    plugin_dir: list[Path] = typer.Option(
        [],
        "--plugin-dir",
        help="Plugin directories to load (can be specified multiple times)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    no_plugins: bool = typer.Option(
        False,
        "--no-plugins",
        help="Disable plugin loading",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with verbose logging",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
) -> None:
    """Start the reasoning-mcp server.

    The run command initializes the MCP server, loads native reasoning methods,
    optionally loads plugins, and starts the server with the specified transport.

    Examples:
        # Start with stdio transport (default for MCP)
        $ reasoning-mcp run

        # Start HTTP server on custom port
        $ reasoning-mcp run --transport http --port 8000

        # Enable debug mode
        $ reasoning-mcp run --debug

        # Load plugins from custom directories
        $ reasoning-mcp run --plugin-dir ./my-plugins --plugin-dir ./other-plugins

        # Disable plugins
        $ reasoning-mcp run --no-plugins
    """
    # Validate transport
    valid_transports = ["stdio", "http", "websocket"]
    if transport.lower() not in valid_transports:
        console.print(
            f"[red]Error:[/red] Invalid transport '{transport}'. "
            f"Must be one of: {', '.join(valid_transports)}"
        )
        raise typer.Exit(1)

    # Normalize transport
    transport = transport.lower()

    # Configure logging
    _configure_logging(log_level if not debug else "DEBUG")

    # Run async startup
    try:
        asyncio.run(_run_server_async(
            host=host,
            port=port,
            transport=transport,
            plugin_dirs=list(plugin_dir) if plugin_dir else [],
            no_plugins=no_plugins,
            debug=debug,
            log_level=log_level if not debug else "DEBUG",
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error starting server:[/red] {e}")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


async def _run_server_async(
    host: str,
    port: int,
    transport: str,
    plugin_dirs: list[Path],
    no_plugins: bool,
    debug: bool,
    log_level: str,
) -> None:
    """Async implementation of server startup.

    Args:
        host: Host to bind to
        port: Port to bind to
        transport: Transport type (stdio, http, websocket)
        plugin_dirs: List of plugin directories
        no_plugins: Whether to disable plugin loading
        debug: Whether debug mode is enabled
        log_level: Logging level string
    """
    from reasoning_mcp.config import Settings
    from reasoning_mcp.methods.native import register_all_native_methods
    from reasoning_mcp.plugins.loader import PluginLoader
    from reasoning_mcp.registry import MethodRegistry
    from reasoning_mcp.server import mcp

    # Create settings (will use env vars + .env file)
    settings = Settings(
        log_level=log_level,  # type: ignore
        debug=debug,
    )

    # Override plugin settings if specified
    if no_plugins:
        settings.enable_plugins = False
    if plugin_dirs:
        # Use the first plugin dir or create a custom loader
        pass  # Plugin dirs will be handled by loader

    console.print("[bold blue]Starting reasoning-mcp server...[/bold blue]")
    console.print(f"Transport: {transport}")
    console.print(f"Debug mode: {'enabled' if debug else 'disabled'}")
    console.print(f"Log level: {log_level}")

    # Initialize registry
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create method registry
        task = progress.add_task("Initializing method registry...", total=None)
        registry = MethodRegistry()
        await registry.initialize()
        progress.update(task, completed=True)

        # Register native methods
        task = progress.add_task("Registering native methods...", total=None)
        results = register_all_native_methods(registry)
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        progress.update(task, completed=True)

        console.print(
            f"[green]Registered {successful} of {total} native methods[/green]"
        )

        # Load plugins (if enabled)
        if settings.enable_plugins and not no_plugins:
            task = progress.add_task("Loading plugins...", total=None)

            # Determine plugin directories
            dirs_to_search = plugin_dirs if plugin_dirs else [settings.plugins_dir]

            # Create loader with registry and settings
            loader = PluginLoader(
                plugin_dirs=dirs_to_search,
                registry=registry,
                settings=settings,
            )

            # Discover plugins
            discovered = await loader.discover()

            if discovered:
                # Load all plugins (they will auto-register with the registry)
                plugins = await loader.load_all()

                console.print(f"[green]Loaded {len(plugins)} plugins[/green]")
            else:
                console.print("[yellow]No plugins discovered[/yellow]")

            progress.update(task, completed=True)
        else:
            console.print("[dim]Plugin loading disabled[/dim]")

    # Display server info
    console.print("\n[bold green]Server initialized successfully![/bold green]")
    console.print(f"Registered methods: {registry.method_count}")

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame) -> None:  # type: ignore
        """Handle shutdown signals."""
        console.print("\n[yellow]Received shutdown signal, stopping server...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server with appropriate transport
    console.print(f"\n[bold]Starting server with {transport} transport...[/bold]")

    try:
        if transport == "stdio":
            console.print("[dim]Server running on stdio (communicate via stdin/stdout)[/dim]")
            # Run the server using stdio transport
            await mcp.run_stdio_async()

        elif transport == "http":
            console.print(f"[dim]Server URL: http://{host}:{port}[/dim]")
            console.print("[cyan]Press Ctrl+C to stop[/cyan]\n")

            # Update FastMCP settings
            mcp.settings.host = host
            mcp.settings.port = port

            # Run using SSE (Server-Sent Events) for HTTP
            await mcp.run_sse_async()

        elif transport == "websocket":
            console.print(f"[dim]Server URL: ws://{host}:{port}[/dim]")
            console.print("[cyan]Press Ctrl+C to stop[/cyan]\n")

            # Update FastMCP settings
            mcp.settings.host = host
            mcp.settings.port = port

            # Run using streamable HTTP (which supports WebSocket)
            await mcp.run_streamable_http_async()

    except asyncio.CancelledError:
        console.print("[yellow]Server cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error:[/red] {e}")
        if debug:
            console.print_exception()
        raise
    finally:
        console.print("[dim]Cleaning up...[/dim]")
        # Cleanup would happen here if needed
        console.print("[green]Server stopped[/green]")


def _configure_logging(level: str) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )

    # Set specific loggers
    logging.getLogger("reasoning_mcp").setLevel(numeric_level)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
