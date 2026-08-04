//go:build integration

package integration

import (
	"context"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropic_Completion(t *testing.T) {
	service := aiwire.NewAnthropicAPIService(keyOrSkip(t, "ANTHROPIC_API_KEY"))
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:           "claude-sonnet-5",
		OmitTemperature: true,
	})
}

func TestAnthropic_Streaming(t *testing.T) {
	service := aiwire.NewAnthropicAPIService(keyOrSkip(t, "ANTHROPIC_API_KEY"))
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:           "claude-sonnet-5",
		OmitTemperature: true,
	})
}

// Claude 5 models take adaptive thinking driven by effort, not a token budget.
// Adaptive returns a signature for replay but no thinking text, and decides for
// itself whether to think at all — so only the request shape is asserted here.
func TestAnthropic_ReasoningEffort(t *testing.T) {
	service := aiwire.NewAnthropicAPIService(keyOrSkip(t, "ANTHROPIC_API_KEY"))
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("What is 127 * 341? Show your working."),
	}

	maxTokens := 8000
	resp, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:     "claude-sonnet-5",
		MaxTokens: &maxTokens,
		Reasoning: &aiwire.ReasoningOption{Effort: aiwire.ReasoningEffortMedium},
	})

	require.NoError(t, err)
	assert.NotEmpty(t, resp.Message.Content)

	t.Logf("Reasoning: %d chars, %d details, %d tokens", len(resp.Reasoning),
		len(resp.ReasoningDetails), resp.Usage.CompletionTokensDetails.ReasoningTokens)
	logUsage(t, resp.Usage)
}

// Effort none must turn thinking off rather than leaving it unset.
func TestAnthropic_ReasoningEffortNone(t *testing.T) {
	service := aiwire.NewAnthropicAPIService(keyOrSkip(t, "ANTHROPIC_API_KEY"))
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("What is 2 + 2?"),
	}

	resp, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:           "claude-sonnet-5",
		OmitTemperature: true,
		Reasoning:       &aiwire.ReasoningOption{Effort: aiwire.ReasoningEffortNone},
	})

	require.NoError(t, err)
	assert.NotEmpty(t, resp.Message.Content)
	assert.Empty(t, resp.ReasoningDetails)

	t.Logf("Content: %s", resp.Message.Content)
}
