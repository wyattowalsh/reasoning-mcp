"""CLI install command for managing plugin installation.

This module provides the `install` command group for installing, uninstalling,
listing, and updating plugins for the reasoning-mcp server.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

from reasoning_mcp.config import get_settings

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext
from reasoning_mcp.plugins.interface import PluginMetadata

console = Console()


class InstallError(Exception):
    """Exception raised when plugin installation fails."""

    pass


def get_plugins_directory() -> Path:
    """Get the plugins directory path from settings.

    Returns:
        Path to the plugins directory.
    """
    settings = get_settings()
    plugins_dir = settings.plugins_dir

    # Ensure directory exists
    plugins_dir.mkdir(parents=True, exist_ok=True)

    return plugins_dir


def validate_plugin_structure(plugin_path: Path) -> PluginMetadata:
    """Validate that a plugin has the required structure and files.

    Args:
        plugin_path: Path to the plugin directory.

    Returns:
        PluginMetadata object with plugin information.

    Raises:
        InstallError: If plugin structure is invalid.
    """
    # Check for plugin.json or plugin.toml
    plugin_json = plugin_path / "plugin.json"
    plugin_toml = plugin_path / "plugin.toml"

    metadata_dict: dict[str, Any] = {}

    if plugin_json.exists():
        try:
            metadata_dict = json.loads(plugin_json.read_text())
        except json.JSONDecodeError as e:
            raise InstallError(f"Invalid plugin.json: {e}") from e
    elif plugin_toml.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]

        try:
            metadata_dict = tomllib.loads(plugin_toml.read_text())
        except Exception as e:
            raise InstallError(f"Invalid plugin.toml: {e}") from e
    else:
        raise InstallError(
            f"Plugin missing metadata file: {plugin_path}\nExpected plugin.json or plugin.toml"
        )

    # Validate required fields
    required_fields = ["name", "version", "author", "description"]
    missing_fields = [field for field in required_fields if field not in metadata_dict]

    if missing_fields:
        raise InstallError(f"Plugin metadata missing required fields: {', '.join(missing_fields)}")

    # Create PluginMetadata object
    return PluginMetadata(
        name=metadata_dict["name"],
        version=metadata_dict["version"],
        author=metadata_dict["author"],
        description=metadata_dict["description"],
        dependencies=metadata_dict.get("dependencies", []),
        entry_point=metadata_dict.get("entry_point", ""),
    )


def install_dependencies(dependencies: list[str]) -> None:
    """Install plugin dependencies using pip.

    Args:
        dependencies: List of dependency specifiers.

    Raises:
        InstallError: If dependency installation fails.
    """
    if not dependencies:
        return

    console.print(f"Installing {len(dependencies)} dependencies...")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + dependencies,
            check=True,
            capture_output=True,
            text=True,
        )
        console.print("[green]✓ Dependencies installed successfully[/green]")
    except subprocess.CalledProcessError as e:
        raise InstallError(f"Failed to install dependencies:\n{e.stderr}") from e


def copy_plugin_files(source: Path, destination: Path) -> None:
    """Copy plugin files to the plugins directory.

    Args:
        source: Source plugin directory.
        destination: Destination in plugins directory.

    Raises:
        InstallError: If copying fails.
    """
    try:
        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(source, destination)
        console.print(f"[green]✓ Plugin files copied to {destination}[/green]")
    except Exception as e:
        raise InstallError(f"Failed to copy plugin files: {e}") from e


def _parse_github_source(source: str) -> tuple[str, str | None]:
    """Parse a GitHub source specifier into URL and optional ref.

    Supports formats:
    - github:user/repo
    - gh:user/repo
    - user/repo (if matches pattern)
    - github:user/repo@branch
    - github:user/repo#tag
    - https://github.com/user/repo

    Args:
        source: GitHub source specifier.

    Returns:
        Tuple of (git_url, ref) where ref is branch/tag or None.
    """
    import re

    ref: str | None = None

    # Handle github: or gh: prefix
    if source.startswith(("github:", "gh:")):
        source = source.split(":", 1)[1]

    # Extract ref from @branch or #tag suffix
    if "@" in source and not source.startswith("http"):
        source, ref = source.rsplit("@", 1)
    elif "#" in source and not source.startswith("http"):
        source, ref = source.rsplit("#", 1)

    # If it's already a full URL, return as-is
    if source.startswith(("http://", "https://", "git@")):
        return source, ref

    # Check if it looks like user/repo format
    if re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", source):
        return f"https://github.com/{source}.git", ref

    return source, ref


def _clone_git_repo(git_url: str, ref: str | None = None) -> Path:
    """Clone a Git repository to a temporary directory.

    Args:
        git_url: Git URL to clone.
        ref: Optional branch, tag, or commit to checkout.

    Returns:
        Path to the cloned repository.

    Raises:
        InstallError: If cloning fails.
    """
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="reasoning-mcp-plugin-"))
    console.print(f"[cyan]Cloning from {git_url}...[/cyan]")

    try:
        # Clone the repository
        clone_cmd = ["git", "clone", "--depth", "1"]
        if ref:
            clone_cmd.extend(["--branch", ref])
        clone_cmd.extend([git_url, str(temp_dir)])

        result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
        )

        # If shallow clone with ref failed, try full clone
        if result.returncode != 0 and ref:
            console.print(f"[yellow]Shallow clone failed, trying full clone...[/yellow]")
            temp_dir.rmdir() if temp_dir.exists() else None
            temp_dir = Path(tempfile.mkdtemp(prefix="reasoning-mcp-plugin-"))

            subprocess.run(
                ["git", "clone", git_url, str(temp_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Checkout the specific ref
            subprocess.run(
                ["git", "checkout", ref],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        elif result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, clone_cmd, result.stdout, result.stderr)

        console.print(f"[green]✓ Cloned successfully[/green]")
        return temp_dir

    except subprocess.CalledProcessError as e:
        raise InstallError(f"Failed to clone repository:\n{e.stderr}") from e
    except FileNotFoundError:
        raise InstallError("Git is not installed or not in PATH") from None


def resolve_source(source: str) -> Path:
    """Resolve plugin source to a local path.

    Supports:
    - Local filesystem paths
    - Git URLs (clones to temporary directory)
    - GitHub shorthand: github:user/repo, gh:user/repo, or user/repo
    - GitHub with ref: github:user/repo@branch or github:user/repo#tag
    - PyPI package names (installs to temporary directory)

    Args:
        source: Plugin source specifier.

    Returns:
        Path to the plugin directory.

    Raises:
        InstallError: If source cannot be resolved.

    Examples:
        resolve_source("./my-plugin")           # Local path
        resolve_source("github:user/repo")      # GitHub shorthand
        resolve_source("gh:user/repo@v1.0")     # GitHub with tag
        resolve_source("user/repo")             # GitHub user/repo format
        resolve_source("my-plugin")             # PyPI package
    """
    import re

    # Check if it's a local path
    local_path = Path(source)
    if local_path.exists() and local_path.is_dir():
        return local_path.resolve()

    # Check for GitHub shorthand or Git URL
    is_github_shorthand = (
        source.startswith(("github:", "gh:"))
        or re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(@|#|$)", source)
    )
    is_git_url = source.startswith(("http://", "https://", "git@"))

    if is_github_shorthand or is_git_url:
        git_url, ref = _parse_github_source(source)
        return _clone_git_repo(git_url, ref)

    # Assume it's a PyPI package name
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="reasoning-mcp-plugin-"))
    console.print(f"[cyan]Installing package {source} from PyPI...[/cyan]")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(temp_dir), source],
            check=True,
            capture_output=True,
            text=True,
        )

        # Find the plugin directory (should contain plugin.json/toml)
        for item in temp_dir.iterdir():
            if item.is_dir() and any(
                (item / name).exists() for name in ["plugin.json", "plugin.toml"]
            ):
                return item

        raise InstallError(f"Package {source} does not appear to be a valid reasoning-mcp plugin")
    except subprocess.CalledProcessError as e:
        raise InstallError(f"Failed to install package:\n{e.stderr}") from e


def install_plugin_impl(ctx: "CLIContext", plugin_name: str, upgrade: bool) -> None:
    """Install a plugin from source.

    Args:
        ctx: CLI context containing settings, logger, and registry.
        plugin_name: Plugin name or path to install.
        upgrade: Whether to upgrade if already installed.

    This is the implementation function called by the main CLI.
    """
    try:
        # Resolve source to local path
        console.print(f"Resolving plugin source: {plugin_name}")
        plugin_source = resolve_source(plugin_name)

        # Validate plugin structure
        console.print("Validating plugin structure...")
        metadata = validate_plugin_structure(plugin_source)

        # Check if plugin already exists
        plugins_dir = get_plugins_directory()
        plugin_dest = plugins_dir / metadata.name

        if plugin_dest.exists() and not upgrade:
            raise InstallError(f"Plugin '{metadata.name}' already exists. Use --upgrade to reinstall.")

        # Display plugin information
        console.print("\n[bold]Plugin Information:[/bold]")
        console.print(f"  Name: {metadata.name}")
        console.print(f"  Version: {metadata.version}")
        console.print(f"  Author: {metadata.author}")
        console.print(f"  Description: {metadata.description}")

        if metadata.dependencies:
            console.print(f"  Dependencies: {', '.join(metadata.dependencies)}")

        # Install dependencies
        if metadata.dependencies:
            install_dependencies(metadata.dependencies)

        # Copy plugin files
        console.print(f"\nInstalling plugin to {plugin_dest}...")
        copy_plugin_files(plugin_source, plugin_dest)

        console.print()
        console.print(f"[bold green]✓ Successfully installed plugin '{metadata.name}'[/bold green]")
        console.print("\nRestart the server for the plugin to take effect.")

    except InstallError as e:
        console.print(f"[bold red]✗ Installation failed: {e}[/bold red]", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Unexpected error: {e}[/bold red]", style="red")
        raise typer.Exit(1)


def list_plugins_impl(ctx: "CLIContext") -> None:
    """Show installed plugins with status.

    Args:
        ctx: CLI context containing settings, logger, and registry.

    This is the implementation function called by the main CLI.
    """
    try:
        plugins_dir = get_plugins_directory()

        if not plugins_dir.exists() or not any(plugins_dir.iterdir()):
            console.print("No plugins installed.")
            return

        console.print("[bold]Installed Plugins:[/bold]\n")

        plugin_count = 0
        for plugin_path in sorted(plugins_dir.iterdir()):
            if not plugin_path.is_dir():
                continue

            try:
                metadata = validate_plugin_structure(plugin_path)
                plugin_count += 1

                console.print(f"  [cyan]{metadata.name}[/cyan] (v{metadata.version})")
                console.print(f"    Description: {metadata.description}")
                console.print(f"    Author: {metadata.author}")
                console.print(f"    Location: {plugin_path}")
                console.print()
            except InstallError as e:
                console.print(f"  [yellow]{plugin_path.name} [INVALID]: {e}[/yellow]")
                console.print()

        if plugin_count == 0:
            console.print("No valid plugins found.")
        else:
            console.print(f"Total: {plugin_count} plugin(s)")

    except Exception as e:
        console.print(f"[bold red]✗ Error listing plugins: {e}[/bold red]")
        raise typer.Exit(1)


def uninstall_plugin_impl(ctx: "CLIContext", plugin_name: str) -> None:
    """Uninstall a plugin and cleanup its data.

    Args:
        ctx: CLI context containing settings, logger, and registry.
        plugin_name: Name of the plugin to uninstall.

    This is the implementation function called by the main CLI.
    """
    try:
        plugins_dir = get_plugins_directory()
        plugin_path = plugins_dir / plugin_name

        if not plugin_path.exists():
            raise InstallError(f"Plugin '{plugin_name}' is not installed")

        # Confirm deletion
        if not typer.confirm(f"Are you sure you want to uninstall '{plugin_name}'?"):
            console.print("Uninstall cancelled.")
            raise typer.Exit(0)

        # Remove plugin directory
        shutil.rmtree(plugin_path)

        console.print(f"[bold green]✓ Successfully uninstalled plugin '{plugin_name}'[/bold green]")
        console.print("Restart the server for changes to take effect.")

    except InstallError as e:
        console.print(f"[bold red]✗ Uninstall failed: {e}[/bold red]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]✗ Unexpected error: {e}[/bold red]")
        raise typer.Exit(1)


__all__ = ["install_plugin_impl", "list_plugins_impl", "uninstall_plugin_impl", "InstallError"]
