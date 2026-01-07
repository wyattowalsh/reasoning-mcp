"""Entry point for python -m reasoning_mcp."""


def main() -> None:
    """Run the reasoning-mcp CLI application."""
    from reasoning_mcp.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
