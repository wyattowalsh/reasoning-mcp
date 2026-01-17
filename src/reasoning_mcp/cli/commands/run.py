"""CLI run command for starting the reasoning-mcp server.

This module provides the 'run' command that starts the MCP server with
configured transport (stdio, HTTP, or WebSocket), loads plugins, and
registers native methods.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import signal
import socket
import sys
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from pathlib import Path

    from reasoning_mcp.cli.main import CLIContext

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host address to check.
        port: Port number to check.

    Returns:
        True if the port is available, False otherwise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _check_required_dependencies() -> list[str]:
    """Check if required dependencies are installed.

    Returns:
        List of missing dependency names.
    """
    missing: list[str] = []

    # Core dependencies
    try:
        import mcp  # noqa: F401
    except ImportError:
        missing.append("mcp")

    try:
        import pydantic  # noqa: F401
    except ImportError:
        missing.append("pydantic")

    try:
        import structlog  # noqa: F401
    except ImportError:
        missing.append("structlog")

    # Optional but commonly needed
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        # Only warn if using SSE/websocket transport
        pass

    return missing


def _run_preflight_checks(host: str, port: int, transport: str, debug: bool) -> None:
    """Run pre-flight checks before starting the server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        transport: Transport type.
        debug: Whether debug mode is enabled.

    Raises:
        typer.Exit: If any pre-flight check fails.
    """
    # Log configuration summary at debug level
    logger.debug(
        "Server configuration: host=%s, port=%d, transport=%s, debug=%s",
        host,
        port,
        transport,
        debug,
    )

    # Check required dependencies
    missing_deps = _check_required_dependencies()
    if missing_deps:
        console.print(f"[red]Error:[/red] Missing required dependencies: {', '.join(missing_deps)}")
        console.print("[dim]Try: uv sync --all-extras[/dim]")
        raise typer.Exit(1)

    # Check port availability for network transports
    if transport in ("sse", "websocket"):
        if not _check_port_available(host, port):
            console.print(f"[red]Error:[/red] Port {port} is already in use on {host}")
            console.print(f"[dim]Try: lsof -i :{port} to see what's using it[/dim]")
            console.print(f"[dim]Or use a different port: --port {port + 1}[/dim]")
            raise typer.Exit(1)

        # Check for privileged port
        if port < 1024:
            console.print(f"[yellow]Warning:[/yellow] Port {port} is a privileged port (<1024)")
            console.print("[dim]You may need elevated privileges to bind to this port[/dim]")

    logger.debug("Pre-flight checks passed")


# ---------------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------------


def _handle_os_error(e: OSError, host: str, port: int, debug: bool) -> None:
    """Handle OS-level errors during server startup.

    Args:
        e: The OSError exception.
        host: Host that was being bound to.
        port: Port that was being bound to.
        debug: Whether debug mode is enabled.
    """
    if e.errno == errno.EADDRINUSE:
        console.print(f"[red]Error:[/red] Port {port} is already in use")
        console.print(f"[dim]Try: lsof -i :{port} to see what's using it[/dim]")
        console.print(f"[dim]Or use a different port: --port {port + 1}[/dim]")
    elif e.errno == errno.EACCES:
        console.print(f"[red]Error:[/red] Permission denied binding to {host}:{port}")
        if port < 1024:
            console.print(
                "[dim]Ports below 1024 require elevated privileges. "
                "Try using a port > 1024 (e.g., --port 8000)[/dim]"
            )
        else:
            console.print("[dim]Try running with elevated privileges or check firewall rules[/dim]")
    elif e.errno == errno.ECONNREFUSED:
        console.print(f"[red]Error:[/red] Connection refused to {host}:{port}")
        console.print("[dim]The server may not be running or the address may be incorrect[/dim]")
    elif e.errno == errno.ENETUNREACH:
        console.print(f"[red]Error:[/red] Network unreachable for {host}")
        console.print("[dim]Check your network connection and try again[/dim]")
    elif e.errno == errno.EADDRNOTAVAIL:
        console.print(f"[red]Error:[/red] Address {host} is not available on this system")
        console.print("[dim]Try using 127.0.0.1 or 0.0.0.0 instead[/dim]")
    else:
        console.print(f"[red]OS Error ({e.errno}):[/red] {e}")
        if debug:
            console.print_exception()

    raise typer.Exit(1)


def _handle_import_error(e: ImportError, debug: bool) -> None:
    """Handle import errors during server startup.

    Args:
        e: The ImportError exception.
        debug: Whether debug mode is enabled.
    """
    module_name = e.name if e.name else "unknown"
    console.print(f"[red]Import Error:[/red] Failed to import '{module_name}'")
    console.print(f"[dim]Details: {e}[/dim]")
    console.print("\n[bold]Suggested fixes:[/bold]")
    console.print("  1. Install missing dependencies: [cyan]uv sync --all-extras[/cyan]")
    console.print("  2. Check if the module name is correct")
    console.print("  3. Verify your Python environment is activated")

    if debug:
        console.print("\n[dim]Full traceback:[/dim]")
        console.print_exception()

    raise typer.Exit(1)


def _handle_permission_error(e: PermissionError, debug: bool) -> None:
    """Handle permission errors during server startup.

    Args:
        e: The PermissionError exception.
        debug: Whether debug mode is enabled.
    """
    console.print(f"[red]Permission Error:[/red] {e}")
    console.print("\n[bold]Suggested fixes:[/bold]")
    console.print("  1. Check file/directory permissions")
    console.print("  2. Ensure you have read/write access to the working directory")
    console.print("  3. Try running with appropriate user privileges")

    if debug:
        console.print_exception()

    raise typer.Exit(1)


def _handle_runtime_error(e: RuntimeError, debug: bool) -> None:
    """Handle runtime errors during server startup.

    Args:
        e: The RuntimeError exception.
        debug: Whether debug mode is enabled.
    """
    error_msg = str(e).lower()

    if "event loop" in error_msg:
        console.print("[red]Error:[/red] Event loop conflict detected")
        console.print(
            "[dim]This may occur when running in certain environments (e.g., Jupyter)[/dim]"
        )
        console.print("[dim]Try running the server from a standard terminal[/dim]")
    elif "uvloop" in error_msg:
        console.print("[red]Error:[/red] uvloop initialization failed")
        console.print("[dim]Try: export UVLOOP_USE_UVLOOP=0[/dim]")
    else:
        console.print(f"[red]Runtime Error:[/red] {e}")

    if debug:
        console.print_exception()

    raise typer.Exit(1)


def _handle_value_error(e: ValueError, debug: bool) -> None:
    """Handle value errors during server startup.

    Args:
        e: The ValueError exception.
        debug: Whether debug mode is enabled.
    """
    console.print(f"[red]Configuration Error:[/red] {e}")
    console.print("[dim]Check your configuration settings and command-line arguments[/dim]")

    if debug:
        console.print_exception()

    raise typer.Exit(1)


def run_server(ctx: CLIContext) -> None:
    """Start the MCP server using the CLI context (legacy function).

    Args:
        ctx: CLI context containing settings, logger, and registry.

    Note:
        This function is kept for backward compatibility.
        Use run_command() for full control over server options.
    """
    run_command(
        ctx=ctx,
        host="127.0.0.1",
        port=8000,
        transport="stdio",
        debug=ctx.settings.debug.enabled,
        reload=False,
    )


def run_command(
    ctx: CLIContext,
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "stdio",
    debug: bool = False,
    reload: bool = False,
) -> None:
    """Start the MCP server with specified options.

    Args:
        ctx: CLI context containing settings, logger, and registry.
        host: Host to bind to (default: 127.0.0.1).
        port: Port to bind to (default: 8000).
        transport: Transport type - stdio, sse, or websocket (default: stdio).
        debug: Enable debug logging (default: False).
        reload: Enable auto-reload for development (default: False).
    """
    # Validate transport
    valid_transports = ["stdio", "sse", "websocket"]
    transport_lower = transport.lower()
    if transport_lower not in valid_transports:
        console.print(
            f"[red]Error:[/red] Invalid transport '{transport}'. "
            f"Must be one of: {', '.join(valid_transports)}"
        )
        raise typer.Exit(1)

    # Warn about reload with stdio
    if reload and transport_lower == "stdio":
        console.print(
            "[yellow]Warning:[/yellow] --reload is not supported with stdio transport. "
            "Use --transport sse or --transport websocket for auto-reload."
        )
        reload = False

    # Determine log level
    log_level = "DEBUG" if debug else ctx.settings.log_level

    # Configure logging
    _configure_logging(log_level)

    # Run pre-flight checks
    _run_preflight_checks(host, port, transport_lower, debug)

    # Run async startup with specific error handlers
    try:
        asyncio.run(
            _run_server_async(
                host=host,
                port=port,
                transport=transport_lower,
                plugin_dirs=[],
                no_plugins=not ctx.settings.enable_plugins,
                debug=debug,
                log_level=log_level,
                reload=reload,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        raise typer.Exit(0)
    except OSError as e:
        _handle_os_error(e, host, port, debug)
    except ImportError as e:
        _handle_import_error(e, debug)
    except PermissionError as e:
        _handle_permission_error(e, debug)
    except RuntimeError as e:
        _handle_runtime_error(e, debug)
    except ValueError as e:
        _handle_value_error(e, debug)
    except Exception as e:
        console.print(f"[red]Error starting server:[/red] {e}")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


def run(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type: stdio, sse, or websocket",
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
        "-d",
        help="Enable debug mode with verbose logging",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development (SSE/WebSocket only)",
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

        # Start SSE server on custom port
        $ reasoning-mcp run --transport sse --port 9000

        # Enable debug mode
        $ reasoning-mcp run --debug

        # Development mode with auto-reload
        $ reasoning-mcp run --transport sse --reload

        # Load plugins from custom directories
        $ reasoning-mcp run --plugin-dir ./my-plugins --plugin-dir ./other-plugins

        # Disable plugins
        $ reasoning-mcp run --no-plugins
    """
    # Validate transport
    valid_transports = ["stdio", "sse", "websocket"]
    if transport.lower() not in valid_transports:
        console.print(
            f"[red]Error:[/red] Invalid transport '{transport}'. "
            f"Must be one of: {', '.join(valid_transports)}"
        )
        raise typer.Exit(1)

    # Normalize transport
    transport = transport.lower()

    # Warn about reload with stdio
    if reload and transport == "stdio":
        console.print(
            "[yellow]Warning:[/yellow] --reload is not supported with stdio transport. "
            "Use --transport sse or --transport websocket for auto-reload."
        )
        reload = False

    # Configure logging
    effective_log_level = log_level if not debug else "DEBUG"
    _configure_logging(effective_log_level)

    # Run pre-flight checks
    _run_preflight_checks(host, port, transport, debug)

    # Check uvicorn availability for network transports with reload
    if reload and transport in ("sse", "websocket"):
        try:
            import uvicorn  # noqa: F401
        except ImportError:
            console.print("[red]Error:[/red] uvicorn is required for --reload with SSE/WebSocket")
            console.print("[dim]Install with: uv add uvicorn[/dim]")
            raise typer.Exit(1)

    # Run async startup with specific error handlers
    try:
        asyncio.run(
            _run_server_async(
                host=host,
                port=port,
                transport=transport,
                plugin_dirs=list(plugin_dir) if plugin_dir else [],
                no_plugins=no_plugins,
                debug=debug,
                log_level=effective_log_level,
                reload=reload,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        raise typer.Exit(0)
    except OSError as e:
        _handle_os_error(e, host, port, debug)
    except ImportError as e:
        _handle_import_error(e, debug)
    except PermissionError as e:
        _handle_permission_error(e, debug)
    except RuntimeError as e:
        _handle_runtime_error(e, debug)
    except ValueError as e:
        _handle_value_error(e, debug)
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
    reload: bool = False,
) -> None:
    """Async implementation of server startup.

    Args:
        host: Host to bind to
        port: Port to bind to
        transport: Transport type (stdio, sse, websocket)
        plugin_dirs: List of plugin directories
        no_plugins: Whether to disable plugin loading
        debug: Whether debug mode is enabled
        log_level: Logging level string
        reload: Whether to enable auto-reload for development
    """
    from reasoning_mcp.config import DebugConfig, Settings
    from reasoning_mcp.methods.native import register_all_native_methods
    from reasoning_mcp.plugins.loader import PluginLoader
    from reasoning_mcp.registry import MethodRegistry
    from reasoning_mcp.server import mcp

    # Create settings (will use env vars + .env file)
    settings = Settings(
        log_level=log_level,  # type: ignore[arg-type]
        debug=DebugConfig(enabled=debug),
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

        console.print(f"[green]Registered {successful} of {total} native methods[/green]")

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

    def signal_handler(sig: int, frame: object) -> None:
        """Handle shutdown signals."""
        console.print("\n[yellow]Received shutdown signal, stopping server...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server with appropriate transport
    console.print(f"\n[bold]Starting server with {transport} transport...[/bold]")
    if reload:
        console.print("[yellow]Auto-reload enabled (development mode)[/yellow]")

    try:
        if transport == "stdio":
            console.print("[dim]Server running on stdio (communicate via stdin/stdout)[/dim]")
            # Run the server using stdio transport
            await mcp.run_stdio_async()

        elif transport == "sse":
            console.print(f"[dim]Server URL: http://{host}:{port}[/dim]")
            console.print("[cyan]Press Ctrl+C to stop[/cyan]\n")

            # Update FastMCP settings
            mcp.settings.host = host
            mcp.settings.port = port

            if reload:
                # Use uvicorn with reload for development
                import uvicorn

                config = uvicorn.Config(
                    app=mcp.sse_app(),
                    host=host,
                    port=port,
                    reload=True,
                    log_level=log_level.lower(),
                )
                server = uvicorn.Server(config)
                await server.serve()
            else:
                # Run using SSE (Server-Sent Events)
                await mcp.run_sse_async()

        elif transport == "websocket":
            console.print(f"[dim]Server URL: ws://{host}:{port}[/dim]")
            console.print("[cyan]Press Ctrl+C to stop[/cyan]\n")

            # Update FastMCP settings
            mcp.settings.host = host
            mcp.settings.port = port

            if reload:
                # Use uvicorn with reload for development
                import uvicorn

                config = uvicorn.Config(
                    app=mcp.streamable_http_app(),
                    host=host,
                    port=port,
                    reload=True,
                    log_level=log_level.lower(),
                )
                server = uvicorn.Server(config)
                await server.serve()
            else:
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
