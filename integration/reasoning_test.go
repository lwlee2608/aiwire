//go:build integration

package integration

import (
	"context"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

type reasoningCase struct {
	model    string
	provider *aiwire.ProviderOption
	summary  string
	// gpt-5 toggles between summary and encrypted-only modes; only reasoning_tokens is stable.
	unreliableReasoningText bool
}

func (c reasoningCase) opts(effort aiwire.ReasoningEffort, exclude bool) aiwire.CompletionOption {
	return aiwire.CompletionOption{
		Model:       c.model,
		Temperature: 0.0,
		Provider:    c.provider,
		Reasoning: &aiwire.ReasoningOption{
			Effort:  effort,
			Exclude: exclude,
			Summary: c.summary,
		},
	}
}

func reasoningMessages() []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step."),
	}
}

func reasoningService(t *testing.T) *aiwire.Service {
	t.Helper()
	return aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
}

func runReasoningBasic(t *testing.T, c reasoningCase) {
	t.Helper()
	response, err := reasoningService(t).Completions(context.Background(), reasoningMessages(), nil, c.opts(aiwire.ReasoningEffortLow, false))

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	if !c.unreliableReasoningText {
		assert.NotEmpty(t, response.Reasoning, "expected reasoning content")
	}
	assert.Greater(t, response.Usage.CompletionTokensDetails.ReasoningTokens, int64(0),
		"expected reasoning tokens to be counted")

	t.Logf("Reasoning: %s", response.Reasoning)
	t.Logf("Content: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func runReasoningExclude(t *testing.T, c reasoningCase) {
	t.Helper()
	response, err := reasoningService(t).Completions(context.Background(), reasoningMessages(), nil, c.opts(aiwire.ReasoningEffortLow, true))

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	assert.Empty(t, response.Reasoning, "reasoning text should be excluded")
	assert.Greater(t, response.Usage.CompletionTokensDetails.ReasoningTokens, int64(0),
		"reasoning tokens should still be counted when excluded")

	t.Logf("Content: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func runReasoningStream(t *testing.T, c reasoningCase) {
	t.Helper()
	var fullContent, fullReasoning string
	var reasoningChunks, contentChunks int
	var finalUsage *aiwire.Usage

	err := reasoningService(t).CompletionsStream(context.Background(), reasoningMessages(), nil, c.opts(aiwire.ReasoningEffortLow, false), func(chunk aiwire.StreamChunk) error {
		if chunk.Done {
			finalUsage = chunk.Usage
			return nil
		}
		if chunk.Reasoning != "" {
			fullReasoning += chunk.Reasoning
			reasoningChunks++
		}
		if chunk.Content != "" {
			fullContent += chunk.Content
			contentChunks++
		}
		return nil
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, fullContent)
	assert.Greater(t, contentChunks, 0)
	if !c.unreliableReasoningText {
		assert.NotEmpty(t, fullReasoning, "expected at least one reasoning chunk")
		assert.Greater(t, reasoningChunks, 0)
	}
	if assert.NotNil(t, finalUsage, "expected usage on final chunk") {
		assert.Greater(t, finalUsage.CompletionTokensDetails.ReasoningTokens, int64(0),
			"expected reasoning tokens to be counted")
		logUsage(t, *finalUsage)
	}

	t.Logf("Reasoning chunks=%d total=%q", reasoningChunks, fullReasoning)
	t.Logf("Content chunks=%d total=%q", contentChunks, fullContent)
}

func runReasoningSuite(t *testing.T, c reasoningCase) {
	t.Helper()
	t.Run("Basic", func(t *testing.T) { runReasoningBasic(t, c) })
	t.Run("Exclude", func(t *testing.T) { runReasoningExclude(t, c) })
	t.Run("Stream", func(t *testing.T) { runReasoningStream(t, c) })
}

func TestReasoning_OpenRouter_Sonnet46(t *testing.T) {
	runReasoningSuite(t, reasoningCase{
		model:    "anthropic/claude-sonnet-4.6",
		provider: &aiwire.ProviderOption{Order: []string{"anthropic"}, AllowFallbacks: false},
	})
}

func TestReasoning_OpenRouter_Opus46(t *testing.T) {
	runReasoningSuite(t, reasoningCase{
		model:    "anthropic/claude-opus-4.6",
		provider: &aiwire.ProviderOption{Order: []string{"anthropic"}, AllowFallbacks: false},
	})
}

func TestReasoning_OpenRouter_KimiK26(t *testing.T) {
	runReasoningSuite(t, reasoningCase{
		model:    "moonshotai/kimi-k2.6",
		provider: &aiwire.ProviderOption{AllowFallbacks: true, Sort: "throughput"},
	})
}

func TestReasoning_OpenRouter_GPT55(t *testing.T) {
	runReasoningSuite(t, reasoningCase{
		model:                   "openai/gpt-5.5",
		provider:                &aiwire.ProviderOption{Order: []string{"openai"}, AllowFallbacks: false},
		summary:                 "detailed",
		unreliableReasoningText: true,
	})
}
