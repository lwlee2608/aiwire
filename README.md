# aiwire

[![MIT License][mit-badge]][mit-url]
[![Go Reference][goref-badge]][goref-url]
[![Build Status][actions-badge]][actions-url]

A Go library for building AI agents with tool-calling support. Works with any OpenAI-compatible API.

## Features

- Simple completion and streaming APIs
- Agentic loop with automatic tool execution
- Pre/post tool callbacks for observability
- Parallel execution of concurrent-safe tool calls within a step
- Token usage tracking across multi-step interactions
- Provider-agnostic via the `Completion` interface
- Image generation and editing via a separate `ImageGeneration` interface (OpenRouter)
- Video generation via a `VideoGeneration` interface (OpenRouter) — blocking `GenerateVideo`, or the `VideoJobs` interface (`SubmitVideo` / `PollVideo`) to drive the wait yourself
- `Decisions` interface for typed decision models like `typesafe/jev-1.13` — choice / score / yes-no answers instead of text (OpenRouter `/api/alpha/decisions`, **experimental**)
- Reasoning capture and replay across loop iterations (OpenRouter — see [docs/reasoning.md](docs/reasoning.md))
- `Responses` interface for OpenAI's `/v1/responses` shape (**experimental** — see note below)

## Install

```bash
go get github.com/lwlee2608/aiwire
```

## Overview

The library has three main components:

- **Service** — connects to an OpenAI-compatible API (e.g. OpenRouter) and handles completions
- **Agent** — orchestrates multi-step interactions, parsing tool calls from the LLM and executing them in a loop
- **Tool** — interface for defining tools the agent can invoke

`AnthropicService` is an alternative `Completion` implementation that speaks
Anthropic's native Messages API, either on AWS Bedrock
(`NewAnthropicService(apiKey, region)`, authenticated with a Bedrock API key) or
the first-party API (`NewAnthropicAPIService(apiKey)`). It supports completions,
streaming, tool calls, structured outputs, and extended thinking. Message
content is text-only — image and file parts are rejected — and model listing is
not available. Thinking control differs by model generation; see
[docs/reasoning.md](docs/reasoning.md).

## Parallel tool calls

When the model calls several tools in one step, the agent runs them at the
same time if every called tool implements `ConcurrentTool` and returns `true`.
`Concurrent() == true` means the tool is safe to run alongside any call,
including another call to itself. A step with any unmarked tool runs one call
at a time, as before.

```go
func (t *searchTool) Concurrent() bool { return true }

agent := aiwire.NewAgent(service, 10)                                   // parallel on (default)
serial := aiwire.NewAgent(service, 10, aiwire.WithParallelTools(false)) // always serial
```

Results, tool messages, `ToolCalls` and `ToolResults` stay in the model's call
order, and callbacks are never called concurrently. In a parallel step every
`PreTool` fires before any `PostTool`, so a post no longer follows its own pre:
pair them by `ID` and time calls with `PostTool.Duration`.

A tool that panics no longer unwinds out of `Execute`. The model gets a short
"Tool <name> crashed" result and the loop continues; `PostTool.Result.Error()`
wraps `ErrToolPanic` with the panic value and stack.

## Experimental: Responses API

`Service.Respond` / `Service.RespondStream` and the surrounding types (`Responses`, `ResponsesOption`, `ResponsesResponse`, `ResponsesStreamChunk`) target the `/v1/responses` endpoint and are in beta. The shape may change without notice. OpenRouter's `/v1/responses` is itself a beta endpoint and runs in stateless mode — `PreviousResponseID` and `Store` are honored by OpenAI and ignored by OpenRouter.

## License

MIT

[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: LICENSE
[goref-badge]: https://pkg.go.dev/badge/github.com/lwlee2608/aiwire.svg
[goref-url]: https://pkg.go.dev/github.com/lwlee2608/aiwire
[actions-badge]: https://github.com/lwlee2608/aiwire/actions/workflows/ci.yml/badge.svg
[actions-url]: https://github.com/lwlee2608/aiwire/actions/workflows/ci.yml
