//go:build integration

package integration

import (
	"context"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

const reasoningModel = "z-ai/glm-4.7"

func reasoningMessages() []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step."),
	}
}

func reasoningOpts(r *aiwire.ReasoningOption) aiwire.CompletionOption {
	return aiwire.CompletionOption{
		Model:       reasoningModel,
		Temperature: 0.0,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: r,
	}
}

func TestReasoning_OpenRouter(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	response, err := service.Completions(context.Background(), reasoningMessages(), nil, reasoningOpts(&aiwire.ReasoningOption{
		Effort: aiwire.ReasoningEffortLow,
	}))

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	assert.NotEmpty(t, response.Reasoning, "expected reasoning content")
	assert.Greater(t, response.Usage.CompletionTokensDetails.ReasoningTokens, int64(0),
		"expected reasoning tokens to be counted")

	t.Logf("Reasoning: %s", response.Reasoning)
	t.Logf("Content: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func TestReasoning_OpenRouter_Exclude(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	response, err := service.Completions(context.Background(), reasoningMessages(), nil, reasoningOpts(&aiwire.ReasoningOption{
		Effort:  aiwire.ReasoningEffortLow,
		Exclude: true,
	}))

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	assert.Empty(t, response.Reasoning, "reasoning text should be excluded")
	assert.Greater(t, response.Usage.CompletionTokensDetails.ReasoningTokens, int64(0),
		"reasoning tokens should still be counted when excluded")

	t.Logf("Content: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func TestReasoning_OpenRouter_Stream(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	var fullContent, fullReasoning string
	var reasoningChunks, contentChunks int
	var finalUsage *aiwire.Usage

	err := service.CompletionsStream(context.Background(), reasoningMessages(), nil, reasoningOpts(&aiwire.ReasoningOption{
		Effort: aiwire.ReasoningEffortLow,
	}), func(chunk aiwire.StreamChunk) error {
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
	assert.NotEmpty(t, fullReasoning, "expected at least one reasoning chunk")
	assert.Greater(t, reasoningChunks, 0)
	assert.Greater(t, contentChunks, 0)
	if assert.NotNil(t, finalUsage, "expected usage on final chunk") {
		assert.Greater(t, finalUsage.CompletionTokensDetails.ReasoningTokens, int64(0),
			"expected reasoning tokens to be counted")
		logUsage(t, *finalUsage)
	}

	t.Logf("Reasoning chunks=%d total=%q", reasoningChunks, fullReasoning)
	t.Logf("Content chunks=%d total=%q", contentChunks, fullContent)
}
