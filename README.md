# aiwire

A Go library for building AI agents with tool-calling support. Works with any OpenAI-compatible API.

## Features

- Simple completion and streaming APIs
- Agentic loop with automatic tool execution
- Pre/post tool callbacks for observability
- Token usage tracking across multi-step interactions
- Provider-agnostic via the `Completion` interface

## Install

```bash
go get github.com/lwlee2608/aiwire
```

## Overview

The library has three main components:

- **Service** — connects to an OpenAI-compatible API (e.g. OpenRouter) and handles completions
- **Agent** — orchestrates multi-step interactions, parsing tool calls from the LLM and executing them in a loop
- **Tool** — interface for defining tools the agent can invoke

## License

MIT
