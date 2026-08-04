# Reasoning support

aiwire's reasoning capture and replay is built on OpenRouter's `reasoning_details` wire format. Calling reasoning models through any other gateway works for the answer, but reasoning content is not captured.

## Provider matrix

Verified by integration tests against live providers, May 2026. "Replay" means an assistant turn carrying reasoning was successfully echoed back into a follow-up call.

### Via OpenRouter (`https://openrouter.ai/api/v1`)

| Model                      | Tokens counted | Reasoning text | Encrypted blob | Streaming | Replay across loop |
| -------------------------- | -------------- | -------------- | -------------- | --------- | ------------------ |
| `anthropic/claude-sonnet-4.6` | yes         | yes            | n/a            | yes       | yes                |
| `anthropic/claude-opus-4.6`   | yes         | yes            | n/a            | yes       | yes                |
| `moonshotai/kimi-k2.6`        | yes         | yes            | n/a            | yes       | yes                |
| `openai/gpt-5.5`              | yes         | summary only*  | yes (~1KB)     | yes       | yes (encrypted)    |

\* gpt-5.5 emits a reasoning summary, not the full chain. The replay-relevant payload is the encrypted blob.

### Direct providers

| Endpoint                                        | Tokens counted | Reasoning text | Encrypted blob | Replay |
| ----------------------------------------------- | -------------- | -------------- | -------------- | ------ |
| OpenAI `https://api.openai.com/v1/chat/completions` | yes        | **no**         | **no**         | n/a    |
| OpenAI `https://api.openai.com/v1/responses`        | yes (out-of-scope, see below) |
| `AnthropicService`, budget thinking (4.5-era models) | **no**\*  | yes            | yes (redacted) | yes    |
| `AnthropicService`, adaptive thinking (Claude 5)     | yes       | **no**\*\*     | n/a            | yes    |

`AnthropicService` translates `thinking` / `redacted_thinking` blocks into the
same `reasoning_details` shape used elsewhere, so replay works in both modes.

\* Bedrock folds thinking tokens into `output_tokens` rather than reporting them
separately, so `ReasoningTokens` reads 0 there.

\*\* Claude 5 returns a thinking block carrying only a signature — no text — so
`CompletionResponse.Reasoning` is empty while `ReasoningDetails` still holds the
signature needed for replay.

## Picking the mechanism

Anthropic splits thinking control across two mutually exclusive mechanisms, by
model generation. Each is rejected by the other's models, so `ReasoningOption`
selects between them and `MaxTokens` wins when both are set:

```
ReasoningOption            wire form                              models
-------------------------------------------------------------------------------
MaxTokens: &n       →  thinking.enabled + budget_tokens      4.5-era (Bedrock)
Effort: high        →  thinking.adaptive + output_config     Claude 5
Effort: none        →  thinking.disabled                     both
```

Sending a budget to a Claude 5 model returns *"thinking.type.enabled is not
supported for this model"*; sending `output_config.effort` to a 4.5-era model on
Bedrock returns *"Extra inputs are not permitted"*. Effort maps straight through
for `low`/`medium`/`high`/`xhigh`; `minimal` floors to `low`.

Both thinking modes pin `temperature` to 1, so `AnthropicService` omits it when
either is active. Claude 5 models reject `temperature` outright — set
`CompletionOption.OmitTemperature` for those regardless of thinking.

## Why OpenRouter only

OpenRouter normalises every upstream's reasoning payload into a single
`reasoning_details` array attached to the assistant message in
`/chat/completions`. aiwire surfaces each detail's `type` and `index`, preserves
the full raw JSON object, and re-attaches that raw payload on replay.

Direct OpenAI exposes reasoning via:

- `/v1/chat/completions` — bills `reasoning_tokens` but **never returns content**
- `/v1/responses` — returns reasoning items with `summary` text and (with `include=reasoning.encrypted_content` + `store=false`) an opaque encrypted blob, but uses a completely different request/response shape

Wiring `/v1/responses` would require a parallel `Service` implementation. It's
intentionally out of scope; if you need direct-OpenAI reasoning replay, use
`openai-go`'s `responses` package directly.

## What gets captured

For supported providers, every successful completion (streaming or not) populates:

- `CompletionResponse.Reasoning` — concatenated reasoning text (when the
  provider exposes it)
- `CompletionResponse.ReasoningDetails` — details with `Type`, `Index`, and raw
  wire bytes preserved for verbatim replay. Provider-specific fields such as
  `text`, `data`, `format`, `id`, `signature`, and future fields live in `Raw`.
- `Usage.CompletionTokensDetails.ReasoningTokens` — billed reasoning tokens

For streaming, `StreamChunk.ReasoningDetails` carries fragments per chunk; the
final `Done` chunk carries the merged result. `text` fragments are concatenated;
all other raw fields are last-write-wins so opaque blobs and signatures are not
corrupted.

## Replay across the agent loop

`Agent.Execute` and `Agent.ExecuteStream` thread captured `ReasoningDetails`
into the assistant message via `AssistantMessageWithReasoning`. This satisfies
provider requirements that prior reasoning be echoed back on subsequent
tool-call turns (notably gpt-5.5 at `effort: high`, which can otherwise reject
follow-up calls).

## Known caveats

- **Provider-specific fields are raw-only.** Read `ReasoningDetail.Raw` if you
  need `text`, `data`, `format`, `id`, `signature`, or future fields.
- **Only `text` concatenates during streaming merge.** Encrypted blobs,
  signatures, IDs, formats, and unknown fields are last-write-wins.
- **Indexless fragments get a synthetic slot.** OpenRouter always sends
  `index`; if a provider ever omits it, distinct fragments won't collapse into
  one entry.
