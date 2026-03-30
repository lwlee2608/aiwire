package aiwire

import (
	"context"
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_Completion_OpenRouter(t *testing.T) {
	t.Skip()

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "moonshotai/kimi-k2.5",
		Temperature: 0.7,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &ReasoningOption{
			Effort: ReasoningEffortLow,
		},
	})
}

func TestOpenAI_Streaming_OpenRouter(t *testing.T) {
	t.Skip()

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "moonshotai/kimi-k2.5",
		Temperature: 0.7,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &ReasoningOption{
			Effort: ReasoningEffortLow,
		},
	})
}

// helper functions

func logUsage(t *testing.T, u openai.CompletionUsage) {
	t.Helper()
	t.Logf("Usage: prompt=%d (cached=%d) completion=%d (reasoning=%d) total=%d",
		u.PromptTokens,
		u.PromptTokensDetails.CachedTokens,
		u.CompletionTokens,
		u.CompletionTokensDetails.ReasoningTokens,
		u.TotalTokens,
	)
}

func runCompletionTest(t *testing.T, service *Service, messages []openai.ChatCompletionMessageParamUnion, opts CompletionOption) {
	ctx := context.Background()
	response, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)

	t.Logf("Completion response: %s", response.Message.Content)
	t.Logf("Reasoning: %s", response.Reasoning)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func runStreamingTest(t *testing.T, service *Service, messages []openai.ChatCompletionMessageParamUnion, opts CompletionOption) {
	ctx := context.Background()
	var fullContent string
	var chunkCount int

	err := service.CompletionsStream(ctx, messages, nil, opts, func(chunk StreamChunk) error {
		chunkCount++
		if chunk.Done {
			t.Logf("Stream finished after %d chunks", chunkCount)
			t.Logf("Provider: %s", chunk.Provider)
			if chunk.Usage != nil {
				logUsage(t, *chunk.Usage)
			}
			return nil
		}
		fullContent += chunk.Content
		if chunk.Reasoning != "" {
			t.Logf("Chunk %d: reasoning=%q", chunkCount, chunk.Reasoning)
		} else if chunk.Content != "" {
			t.Logf("Chunk %d: content=%q", chunkCount, chunk.Content)
		}
		return nil
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, fullContent)
	assert.Greater(t, chunkCount, 0)
	t.Logf("Full streaming response: %s", fullContent)
}
