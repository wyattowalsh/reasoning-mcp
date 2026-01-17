<div align="center">

<img src="assets/logo.png" alt="reasoning-mcp logo" width="120" height="120">

# AGENTS.md

</div>

> AI coding agent instructions for **reasoning-mcp**. Human docs: [README.md](./README.md)

---

## Project Overview
<!-- agents-md:auto -->

**reasoning-mcp** is a unified Model Context Protocol (MCP) server that aggregates, normalizes, and orchestrates 100+ advanced reasoning and thinking methodologies through a universal interface. Built with Python and FastMCP v2, it enables AI assistants to leverage sophisticated reasoning patterns including chain-of-thought, tree-of-thought, analogical reasoning, metacognitive reflection, and many more.

- **Type**: MCP server / library
- **Languages**: Python 3.12+
- **License**: MIT
- **Package Manager**: uv
- **Logo**: `assets/logo.png`

### Key Capabilities

- 100+ native reasoning methods exposed as MCP tools
- Agentic composition and orchestration engine
- Interactive reasoning via user elicitation
- Background task support for long-running operations
- Plugin architecture for extensibility
- Full type safety with Pydantic v2

---

## Quick Reference
<!-- agents-md:auto -->

| Task | Command |
|------|---------|
| **Install** | `uv sync --all-extras` |
| **Test** | `uv run pytest` |
| **Test + Coverage** | `uv run pytest --cov --cov-report=html` |
| **Lint** | `uv run ruff check .` |
| **Lint + Fix** | `uv run ruff check . --fix` |
| **Format** | `uv run ruff format .` |
| **Type Check** | `uv run mypy src/` |
| **Pre-commit** | `pre-commit run --all-files` |
| **Run Server** | `uv run reasoning-mcp` |

---

## Setup Commands
<!-- agents-md:auto -->

```bash
# Clone and enter directory
git clone <repo-url> && cd reasoning-mcp

# Install all dependencies (including dev extras)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run pytest -x
```

---

## Technology Stack
<!-- agents-md:auto -->

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Runtime** | Python | 3.12+ | Core language |
| **MCP Framework** | FastMCP | 2.0+ | MCP server implementation |
| **Type System** | Pydantic | 2.x | Data validation and settings |
| **Package Manager** | uv | Latest | Fast dependency management |
| **Linter/Formatter** | Ruff | 0.6+ | Code quality and formatting |
| **Type Checker** | mypy | 1.11+ | Static type checking (strict mode) |
| **Testing** | pytest | 8.0+ | Unit and integration tests |
| **Async Testing** | pytest-asyncio | 0.24+ | Async test support (auto mode) |
| **Coverage** | pytest-cov | 5.0+ | Code coverage reporting |
| **CLI** | Typer | 0.21+ | Command-line interface |
| **HTTP Client** | httpx | 0.27+ | Async HTTP requests |
| **Logging** | structlog | 24.0+ | Structured logging |
| **Pre-commit** | pre-commit | 3.8+ | Git hooks automation |

---

## Project Structure
<!-- agents-md:auto -->

```
reasoning-mcp/
├── src/
│   └── reasoning_mcp/              # Main package (use underscores in imports)
│       ├── __init__.py             # Package initialization
│       ├── __main__.py             # Entry point for python -m
│       ├── py.typed                # PEP 561 type marker
│       ├── config.py               # Configuration management
│       ├── logging.py              # Structured logging setup
│       ├── registry.py             # Method registry
│       ├── selector.py             # Method selection logic
│       ├── sessions.py             # Session management
│       ├── server.py               # FastMCP server implementation
│       ├── cli/                    # CLI commands (Typer)
│       │   ├── main.py             # CLI entry point
│       │   └── commands/           # Subcommands
│       ├── engine/                 # Pipeline execution engine
│       │   ├── executor.py         # Base executor
│       │   ├── conditional.py      # Conditional branching
│       │   ├── loop.py             # Loop constructs
│       │   ├── parallel.py         # Parallel execution
│       │   ├── sequence.py         # Sequential execution
│       │   ├── switch.py           # Switch/case logic
│       │   ├── method.py           # Method execution
│       │   └── registry.py         # Engine-specific registry
│       ├── methods/                # Reasoning method implementations
│       │   ├── base.py             # Base reasoning method class
│       │   └── native/             # Built-in reasoning methods
│       ├── models/                 # Pydantic data models
│       │   ├── core.py             # Core shared models
│       │   ├── thought.py          # Thought/step models
│       │   ├── session.py          # Session models
│       │   ├── pipeline.py         # Pipeline models
│       │   └── tools.py            # Tool input/output models
│       ├── plugins/                # Plugin architecture
│       │   ├── interface.py        # Plugin interface definition
│       │   └── loader.py           # Plugin discovery and loading
│       ├── prompts/                # Prompt templates
│       │   ├── guided.py           # Guided reasoning prompts
│       │   └── pipelines.py        # Pipeline prompts
│       ├── resources/              # MCP resources
│       │   ├── method.py           # Method resource handlers
│       │   ├── session.py          # Session resource handlers
│       │   ├── template.py         # Template resources
│       │   └── trace.py            # Trace/debug resources
│       └── tools/                  # MCP tools
│           ├── compose.py          # Composition tools
│           ├── evaluate.py         # Evaluation tools
│           ├── methods.py          # Method invocation tools
│           ├── reason.py           # Core reasoning tools
│           ├── register.py         # Tool registration utilities
│           └── session.py          # Session management tools
├── tests/                          # Test suite (mirrors src structure)
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── e2e/                        # End-to-end tests
│   └── benchmarks/                 # Performance benchmarks
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── pyproject.toml                  # Project metadata and tool configuration
├── .pre-commit-config.yaml         # Pre-commit hooks
├── uv.lock                         # Dependency lockfile
├── README.md                       # User-facing documentation
├── AGENTS.md                       # This file (AI agent instructions)
└── LICENSE                         # MIT license
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/reasoning_mcp/` | All production code (use underscores in imports) |
| `src/reasoning_mcp/engine/` | Pipeline execution engine (sequence, parallel, conditionals, loops) |
| `src/reasoning_mcp/methods/` | Reasoning method implementations |
| `src/reasoning_mcp/models/` | Pydantic data models |
| `src/reasoning_mcp/plugins/` | Plugin interface and loader |
| `src/reasoning_mcp/tools/` | MCP tool definitions |
| `src/reasoning_mcp/resources/` | MCP resource handlers |
| `tests/` | Test suite mirroring src structure |

### Naming Convention

- **Package name**: `reasoning_mcp` (underscores) for Python imports
- **PyPI name**: `reasoning-mcp` (hyphens) for installation

---

## Testing Instructions
<!-- agents-md:auto -->

### Test Configuration

- **Framework**: pytest with pytest-asyncio
- **Async Mode**: `auto` (no explicit `@pytest.mark.asyncio` needed)
- **Coverage Target**: 90% minimum, 95%+ goal

### Commands

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/unit/test_methods.py

# Run tests matching a pattern
uv run pytest -k "test_chain_of_thought"

# Run tests in parallel (faster)
uv run pytest -n auto

# Exclude slow/integration tests
uv run pytest -m "not slow and not integration"

# Verbose output with locals
uv run pytest -vv --showlocals
```

### Test Organization

```python
# tests/unit/test_example.py
import pytest
from reasoning_mcp.methods import ChainOfThought

class TestChainOfThought:
    """Test suite for Chain of Thought reasoning."""

    def test_basic_reasoning(self) -> None:
        """Test basic COT functionality."""
        # Arrange
        cot = ChainOfThought()

        # Act
        result = cot.reason("What is 2+2?")

        # Assert
        assert result.answer == "4"

    async def test_async_reasoning(self) -> None:
        """Test async COT functionality (auto mode - no marker needed)."""
        cot = ChainOfThought()
        result = await cot.reason_async("Complex query")
        assert result is not None
```

### Test Markers

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.integration   # Integration tests (external deps)
```

### Best Practices

1. Follow **Arrange-Act-Assert** pattern
2. Use descriptive names: `test_<what>_<condition>_<expected>`
3. One assertion per test when possible
4. Use fixtures for shared setup (see `tests/conftest.py`)
5. Mock external dependencies
6. Test both success and error paths

---

## Code Conventions

### Ruff Configuration

- **Line length**: 100 characters
- **Target**: Python 3.12+
- **Enabled rules**: E, F, W, I (isort), UP, B (bugbear), SIM, TCH
- **Quote style**: Double quotes (`"string"`)
- **Indentation**: 4 spaces

### Type Checking (mypy)

- **Mode**: Strict
- **Required**: All functions must have type hints
- **No bare generics**: Use `list[str]` not `list`
- **Exception**: Tests can omit type annotations

### Import Organization

```python
# Standard library
import asyncio
from typing import Any

# Third-party
from pydantic import BaseModel, Field
import fastmcp

# First-party (local)
from reasoning_mcp.models import ReasoningRequest
from reasoning_mcp.utils import logger
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | `lowercase_underscores` | `chain_of_thought.py` |
| Classes | `PascalCase` | `class ReasoningMethod` |
| Functions | `snake_case` | `def process_request()` |
| Constants | `UPPER_CASE` | `MAX_RETRIES = 3` |
| Test files | `test_*.py` | `test_methods.py` |

---

## CI/CD Context
<!-- agents-md:auto -->

### Platform

- **CI**: GitHub Actions
- **Workflow**: `.github/workflows/ci.yml`
- **Workflow Name**: `CI`
- **Triggers**: Push to `main`, Pull Requests to `main`

### Pipeline Jobs

| Job | Purpose | Key Steps |
|-----|---------|-----------|
| `lint` | Code quality | `ruff check .` + `ruff format --check .` |
| `typecheck` | Type safety | `mypy src/` (strict mode) |
| `test` | Tests & coverage | `pytest --cov` + artifact upload |

### Coverage Artifacts

The test job uploads coverage reports:
- **Artifact name**: `coverage-report`
- **Contents**: `coverage.xml`, `htmlcov/`
- **Retention**: 30 days

### Required Checks

All three jobs must pass before merging:
1. `uv run ruff check .` - No lint errors
2. `uv run ruff format --check .` - Code is formatted
3. `uv run mypy src/` - No type errors
4. `uv run pytest --cov` - Tests pass with coverage

---

## Pre-commit Hooks

Automatically run on `git commit`:

1. **trailing-whitespace**: Remove trailing whitespace
2. **end-of-file-fixer**: Ensure files end with newline
3. **check-yaml**: Validate YAML syntax
4. **check-added-large-files**: Prevent large file commits
5. **check-merge-conflict**: Detect merge conflict markers
6. **ruff**: Lint and auto-fix
7. **ruff-format**: Format code
8. **mypy**: Type check

### Skip Hooks (use sparingly)

```bash
git commit --no-verify
```

---

## FastMCP v2 Patterns

### Tool Definition

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("reasoning-mcp")

class ReasoningInput(BaseModel):
    query: str
    method: str

@mcp.tool()
async def reason(input: ReasoningInput) -> dict:
    """Perform reasoning using specified method."""
    return {"result": "..."}
```

### Context Usage

```python
from mcp.server.fastmcp import Context

@mcp.tool()
async def tool_with_context(query: str, ctx: Context) -> dict:
    # Access request state
    user_id = ctx.get_state("user_id")

    # Sample LLM
    response = await ctx.sample("Analyze this query", tools=[...])

    # Elicit user input
    confirmation = await ctx.elicit("Proceed?")

    return {"result": "..."}
```

---

## Common Tasks

### Adding a New Reasoning Method

1. Create method class in `src/reasoning_mcp/methods/native/`
2. Inherit from base `ReasoningMethod` class
3. Implement required abstract methods
4. Add Pydantic models for input/output in `models/`
5. Register in method registry (`registry.py`)
6. Add corresponding tests in `tests/unit/methods/`
7. Update documentation if needed

### Adding a New MCP Tool

1. Define tool function in appropriate `src/reasoning_mcp/tools/` module
2. Use `@mcp.tool()` decorator
3. Add Pydantic model for parameters
4. Return type automatically generates `output_schema`
5. Add tests in `tests/unit/tools/`
6. Document in docstring (used for tool description)

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --all-extras
```

---

## Security Considerations

- **Never commit secrets**: No API keys, passwords, or credentials in code
- **Environment variables**: Use `.env` files (gitignored) for local secrets
- **Input validation**: All external input validated via Pydantic models
- **No eval/exec**: Never use `eval()` or `exec()` on user input
- **Dependency auditing**: Review new dependencies before adding

---

## PR/Commit Guidelines

### Commit Messages

Use conventional commits format:
- `feat: Add new reasoning method`
- `fix: Resolve type error in pipeline`
- `docs: Update AGENTS.md`
- `test: Add coverage for selector`
- `refactor: Simplify session management`

### Before Creating PR

```bash
# Run full quality check
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
uv run pytest --cov
```

### PR Requirements

- All CI checks pass
- Tests added for new functionality
- No decrease in coverage
- Documentation updated if needed

---

## Gotchas & Edge Cases

### Package Import vs Install Name

```python
# Correct: use underscores in imports
from reasoning_mcp.methods import ChainOfThought

# Wrong: don't use hyphens
from reasoning-mcp.methods import ...  # SyntaxError!
```

### Async Test Mode

Tests are configured with `asyncio_mode = "auto"`. You do NOT need `@pytest.mark.asyncio`:

```python
# Correct: just define async test
async def test_async_feature(self):
    result = await some_async_function()
    assert result

# Unnecessary: marker is auto-applied
@pytest.mark.asyncio  # Not needed!
async def test_async_feature(self):
    ...
```

### mypy Strict Mode

All functions must have complete type hints:

```python
# Correct
def process(data: dict[str, Any]) -> list[str]:
    ...

# Wrong - will fail mypy
def process(data):  # Missing type hints
    ...
```

### Pre-commit Hook Failures

If hooks fail, they often auto-fix. Stage the fixes and commit again:

```bash
git add .
git commit -m "your message"  # Hooks run, may auto-fix
git add .                      # Stage auto-fixes
git commit -m "your message"  # Should pass now
```

---

## Additional Resources

- **FastMCP Docs**: https://github.com/jlowin/fastmcp
- **MCP Specification**: https://modelcontextprotocol.io/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **Ruff Rules**: https://docs.astral.sh/ruff/rules/
- **Project PRD**: `reasoning-mcp-prd-final.md`
- **Task List**: `reasoning-mcp-task-list.md`

---

## Questions?

1. Check existing issues and PRs
2. Review the PRD: `reasoning-mcp-prd-final.md`
3. Review task list: `reasoning-mcp-task-list.md`
4. Run tests to ensure changes don't break existing functionality
5. When in doubt, ask for clarification before making architectural changes

---

**Last Updated**: 2026-01-07
**Format Version**: AAIF 1.0 (AI Agent Instruction Format)
**Synced with agents-md-manager**: Yes
