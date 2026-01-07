# reasoning-mcp

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/wyattowalsh/reasoning-mcp)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/wyattowalsh/reasoning-mcp)

**A unified Model Context Protocol (MCP) server that aggregates 30+ advanced reasoning methodologies into a single, powerful interface.**

## Overview

`reasoning-mcp` is an open-source Python MCP server built on FastMCP v2.14+ that provides a comprehensive suite of reasoning capabilities for AI assistants. Instead of managing multiple fragmented reasoning servers, `reasoning-mcp` offers a single integration point with intelligent method selection, composable pipelines, and production-ready features.

### Key Features

- **30 Native Reasoning Methods**: From Chain of Thought to Tree of Thoughts, ReAct, Shannon Thinking, and more
- **Intelligent Auto-Selection**: Automatically chooses the best reasoning method based on query characteristics
- **Composable Pipelines**: Chain multiple reasoning methods together with sequential, parallel, conditional, and loop execution
- **Background Tasks**: Long-running reasoning chains with real-time progress reporting
- **Session Management**: Branching, revision, and merge capabilities for complex reasoning workflows
- **Plugin Architecture**: Extend with custom reasoning methods via standard Python entry points
- **Production Features**: Middleware, authentication, rate limiting, observability, and audit trails
- **Type-Safe**: Fully typed with Pydantic models and mypy strict mode

### Use Cases

- **AI Application Development**: Integrate sophisticated reasoning into your products with a single MCP server
- **Research & Experimentation**: Compare reasoning methods scientifically with standardized interfaces and traces
- **Code Analysis**: Specialized methods for debugging, code review, and software architecture decisions
- **Ethical Decision Making**: Multi-framework ethical analysis with stakeholder consideration
- **Complex Problem Solving**: Compose methods into pipelines for multi-stage analysis

## Installation

### Using pip

```bash
pip install reasoning-mcp
```

### Using uv (recommended)

```bash
uv add reasoning-mcp
```

### From source

```bash
git clone https://github.com/wyattowalsh/reasoning-mcp.git
cd reasoning-mcp
uv sync
```

## Quick Start

### Starting the Server

```bash
# Start with stdio transport (for Claude Desktop)
reasoning-mcp run

# Start with HTTP transport
reasoning-mcp run --transport streamable-http --port 8000

# Start with SSE transport
reasoning-mcp run --transport sse --host 0.0.0.0 --port 8080
```

### Basic Reasoning Example

```python
# Using the MCP client in Python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the server
server_params = StdioServerParameters(
    command="reasoning-mcp",
    args=["run"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # Call the reason tool
        result = await session.call_tool(
            "reason",
            {
                "query": "How can I optimize this sorting algorithm?",
                "hints": {
                    "domain": "code",
                    "complexity": "medium"
                }
            }
        )

        print(result)
```

### Using with Claude Desktop

1. Install the server:
   ```bash
   pip install reasoning-mcp
   ```

2. Configure Claude Desktop by adding to `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "reasoning": {
         "command": "reasoning-mcp",
         "args": ["run"]
       }
     }
   }
   ```

3. Restart Claude Desktop and start using reasoning methods!

### Using the CLI Helper

```bash
# Auto-configure Claude Desktop
reasoning-mcp install --client claude-desktop

# Inspect available methods
reasoning-mcp inspect --methods

# Check health of all methods
reasoning-mcp health
```

## Available Methods

`reasoning-mcp` provides 30 built-in reasoning methods organized into five categories:

### Core Methods (5)

| Method | Identifier | Description |
|--------|-----------|-------------|
| **Sequential Thinking** | `sequential_thinking` | Step-by-step reasoning with explicit thought progression and branching |
| **Chain of Thought** | `chain_of_thought` | Classic step-by-step reasoning with intermediate steps |
| **Tree of Thoughts** | `tree_of_thoughts` | Explore multiple reasoning paths in a tree structure |
| **ReAct** | `react` | Reasoning and Acting - interleave thoughts with tool actions |
| **Self-Consistency** | `self_consistency` | Generate multiple paths and select the most consistent answer |

### High-Value Methods (7)

| Method | Identifier | Description |
|--------|-----------|-------------|
| **Ethical Reasoning** | `ethical_reasoning` | Multi-framework ethical analysis (utilitarian, deontological, virtue, care) |
| **Code Reasoning** | `code_reasoning` | Specialized reasoning for code analysis, debugging, and development |
| **Dialectic** | `dialectic` | Thesis-antithesis-synthesis reasoning for balanced analysis |
| **Shannon Thinking** | `shannon_thinking` | Information-theoretic reasoning with entropy and uncertainty analysis |
| **Self-Reflection** | `self_reflection` | Metacognitive reasoning with self-critique and improvement |
| **Graph of Thoughts** | `graph_of_thoughts` | Graph-based reasoning with nodes and edges |
| **Monte Carlo Tree Search** | `mcts` | Decision-making with exploration-exploitation balance |

### Specialized Methods (10)

| Method | Identifier | Description |
|--------|-----------|-------------|
| **Skeleton of Thought** | `skeleton_of_thought` | Create high-level skeleton first, then fill in details |
| **Least to Most** | `least_to_most` | Break problem into subproblems, solve from simplest to complex |
| **Step Back** | `step_back` | Step back to consider higher-level concepts before solving |
| **Self-Ask** | `self_ask` | Decompose questions into subquestions and answer iteratively |
| **Decomposed Prompting** | `decomposed_prompting` | Break complex tasks into smaller, manageable subtasks |
| **Mathematical Reasoning** | `mathematical_reasoning` | Formal mathematical reasoning with proofs and symbolic manipulation |
| **Abductive** | `abductive` | Inference to the best explanation from observations |
| **Analogical** | `analogical` | Reasoning by analogy to similar problems or situations |
| **Causal Reasoning** | `causal_reasoning` | Analyze cause-effect relationships and causal chains |
| **Socratic** | `socratic` | Question-driven reasoning to uncover assumptions |

### Advanced Methods (3)

| Method | Identifier | Description |
|--------|-----------|-------------|
| **Counterfactual** | `counterfactual` | Explore alternative scenarios and what-if reasoning |
| **Metacognitive** | `metacognitive` | Thinking about thinking - analyze and optimize reasoning processes |
| **Beam Search** | `beam_search` | Maintain multiple promising reasoning paths simultaneously |

### Holistic Methods (5)

| Method | Identifier | Description |
|--------|-----------|-------------|
| **Lateral Thinking** | `lateral_thinking` | Creative, non-linear reasoning to find novel solutions |
| **Lotus Wisdom** | `lotus_wisdom` | Layered insights and interconnected understanding (inspired by Buddhist philosophy) |
| **Atom of Thoughts** | `atom_of_thoughts` | Atomic, composable thought units that can be recombined |
| **Cascade Thinking** | `cascade_thinking` | Progressive refinement through cascading reasoning stages |
| **CRASH** | `crash` | Compact Reasoning And Self-correction Heuristic |

## Pipeline System

The `compose` tool allows you to chain multiple reasoning methods together into sophisticated pipelines:

### Pipeline Composition

```python
# Example: Deep code analysis pipeline
pipeline = {
    "type": "sequence",
    "stages": [
        {
            "type": "method",
            "method": "code_reasoning",
            "config": {"phase": "analyze"}
        },
        {
            "type": "parallel",
            "branches": [
                {
                    "type": "method",
                    "method": "abductive",
                    "config": {"focus": "bug_causes"}
                },
                {
                    "type": "method",
                    "method": "analogical",
                    "config": {"domain": "design_patterns"}
                }
            ],
            "merge_strategy": "synthesize"
        },
        {
            "type": "method",
            "method": "self_reflection",
            "config": {"critique_depth": "high"}
        }
    ]
}

result = await session.call_tool("compose", {
    "pipeline": pipeline,
    "input": "Analyze this authentication bug",
    "trace_level": "verbose"
})
```

### Example Pipeline JSON

```json
{
  "type": "sequence",
  "stages": [
    {
      "type": "method",
      "method": "step_back",
      "transform": {
        "type": "template",
        "template": "Consider the high-level context: {input}"
      }
    },
    {
      "type": "method",
      "method": "tree_of_thoughts"
    },
    {
      "type": "conditional",
      "condition": {
        "type": "confidence",
        "threshold": 0.8
      },
      "then_stage": {
        "type": "method",
        "method": "self_consistency"
      },
      "else_stage": {
        "type": "method",
        "method": "self_reflection"
      }
    }
  ]
}
```

## CLI Commands

### `reasoning-mcp run`

Start the MCP server with various transport options.

```bash
# Stdio (default, for Claude Desktop)
reasoning-mcp run

# HTTP
reasoning-mcp run --transport streamable-http --port 8000

# SSE
reasoning-mcp run --transport sse --host 0.0.0.0 --port 8080

# Debug mode
reasoning-mcp run --debug
```

### `reasoning-mcp inspect`

Inspect server components and capabilities.

```bash
# List all available methods
reasoning-mcp inspect --methods

# List all tools
reasoning-mcp inspect --tools

# List all resources
reasoning-mcp inspect --resources

# List all prompts
reasoning-mcp inspect --prompts
```

### `reasoning-mcp health`

Check the health status of all reasoning methods.

```bash
reasoning-mcp health
```

### `reasoning-mcp install`

Auto-configure MCP clients.

```bash
# Configure Claude Desktop
reasoning-mcp install --client claude-desktop

# Configure Cursor
reasoning-mcp install --client cursor

# Configure VS Code
reasoning-mcp install --client vscode
```

## Configuration

### Environment Variables

```bash
# Server configuration
REASONING_MCP_NAME="reasoning-mcp"
REASONING_MCP_VERSION="0.1.0"
REASONING_MCP_DEBUG=false

# Transport configuration
REASONING_MCP_TRANSPORT="stdio"
REASONING_MCP_HOST="localhost"
REASONING_MCP_PORT=8000

# Background tasks (Docket)
FASTMCP_DOCKET_URL="http://localhost:3141"

# Telemetry
REASONING_MCP_TELEMETRY_ENABLED=true
REASONING_MCP_TELEMETRY_LOG_LEVEL="INFO"

# Rate limiting
REASONING_MCP_RATE_LIMIT_RPS=10
REASONING_MCP_RATE_LIMIT_BURST=20

# Authentication (optional)
REASONING_MCP_AUTH_ENABLED=false
REASONING_MCP_AUTH_ISSUER="https://your-auth-provider.com"
REASONING_MCP_AUTH_AUDIENCE="reasoning-mcp"

# Sampling (for agentic loops)
REASONING_MCP_SAMPLING_PROVIDER="anthropic"
REASONING_MCP_SAMPLING_MODEL="claude-sonnet-4"
```

### Config File

You can also use a `reasoning_mcp.toml` configuration file:

```toml
[server]
name = "reasoning-mcp"
debug = false

[transport]
type = "stdio"
host = "localhost"
port = 8000

[telemetry]
enabled = true
log_level = "INFO"

[rate_limit]
requests_per_second = 10
burst = 20

[sampling]
provider = "anthropic"
model = "claude-sonnet-4"
```

## Plugin Development

Extend `reasoning-mcp` with custom reasoning methods:

### Creating a Plugin

1. **Create your method class**:

```python
# my_plugin/method.py
from reasoning_mcp.methods.base import ReasoningMethod
from reasoning_mcp.models import MethodIdentifier, ThoughtNode, Session
from fastmcp import Context

class MyCustomMethod(ReasoningMethod):
    @property
    def identifier(self) -> MethodIdentifier:
        return MethodIdentifier("my_custom_method")

    @property
    def category(self) -> str:
        return "specialized"

    @property
    def capabilities(self) -> set[str]:
        return {"custom", "experimental"}

    async def execute(
        self,
        query: str,
        session: Session,
        ctx: Context
    ) -> ThoughtNode:
        # Your reasoning logic here
        return ThoughtNode(
            content="My custom reasoning result",
            thought_type="conclusion",
            step_number=1
        )

    async def health_check(self) -> bool:
        return True
```

2. **Create plugin metadata**:

```python
# my_plugin/__init__.py
from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models import MethodIdentifier, MethodCategory

MY_CUSTOM_METADATA = MethodMetadata(
    identifier=MethodIdentifier("my_custom_method"),
    category=MethodCategory.SPECIALIZED,
    is_native=False,
    capabilities={"custom", "experimental"},
    tags=["experimental", "plugin"],
    description="My custom reasoning method"
)
```

3. **Register via entry points** in `pyproject.toml`:

```toml
[project.entry-points."reasoning_mcp.plugins"]
my_plugin = "my_plugin:MyCustomPlugin"
```

4. **Install and use**:

```bash
pip install my-reasoning-plugin
reasoning-mcp health  # Your method should appear!
```

For more details, see the [Plugin Development Guide](docs/plugins.md).

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/wyattowalsh/reasoning-mcp.git
cd reasoning-mcp

# Install dependencies with uv
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/reasoning_mcp --cov-report=html

# Run specific test file
pytest tests/unit/test_registry.py

# Run integration tests
pytest tests/integration/

# Run with parallel execution
pytest -n auto
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/

# Run all checks (runs automatically in CI)
pre-commit run --all-files
```

### Pull Request Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality (aim for >80% coverage)
3. **Update documentation** if adding new features
4. **Follow code style**: We use `ruff` for formatting and linting
5. **Ensure CI passes**: All tests, linting, and type checks must pass
6. **Write clear commit messages**: Use conventional commits format
7. **Update CHANGELOG.md** with your changes

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/my-new-method

# Make your changes and test
pytest tests/

# Format and lint
ruff format . && ruff check .

# Commit with conventional commits
git commit -m "feat: add new reasoning method"

# Push and create PR
git push origin feature/my-new-method
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Resources

- **Documentation**: [docs/](docs/)
- **GitHub Repository**: https://github.com/wyattowalsh/reasoning-mcp
- **PyPI Package**: https://pypi.org/project/reasoning-mcp/
- **Issue Tracker**: https://github.com/wyattowalsh/reasoning-mcp/issues
- **Discussions**: https://github.com/wyattowalsh/reasoning-mcp/discussions

## Related Projects

- **FastMCP**: https://github.com/jlowin/fastmcp
- **Model Context Protocol**: https://modelcontextprotocol.io/
- **Sequential Thinking MCP**: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
- **Shannon Thinking**: https://github.com/shannonthinking/server-shannon-thinking

## Citation

If you use `reasoning-mcp` in your research, please cite:

```bibtex
@software{reasoning_mcp,
  title = {reasoning-mcp: A Unified MCP Server for Advanced Reasoning},
  author = {Walsh, Wyatt},
  year = {2026},
  url = {https://github.com/wyattowalsh/reasoning-mcp}
}
```

---

**Built with [FastMCP](https://github.com/jlowin/fastmcp) and the [Model Context Protocol](https://modelcontextprotocol.io/)**
