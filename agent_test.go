package aiwire

import (
	"context"
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func newTestAgent(t *testing.T) *Agent {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	return NewAgent(service, 5)
}

func testCompletionOption() CompletionOption {
	return CompletionOption{
		Model:       "moonshotai/kimi-k2.5",
		Temperature: 0.7,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &ReasoningOption{
			Effort: ReasoningEffortLow,
		},
	}
}

func TestAgent_Execute_OpenRouter(t *testing.T) {
	t.Skip()

	agent := newTestAgent(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	result, err := agent.Execute(context.Background(), messages, nil, testCompletionOption(), nil, nil)
	assert.NoError(t, err)
	assert.NotEmpty(t, result.Content)
	assert.NotEmpty(t, result.Provider)

	t.Logf("Content: %s", result.Content)
	t.Logf("Provider: %s", result.Provider)
	logUsage(t, result.Usage)
}

func TestAgent_ExecuteStream_OpenRouter(t *testing.T) {
	t.Skip()

	agent := newTestAgent(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	var fullContent string
	var chunkCount int

	result, err := agent.ExecuteStream(context.Background(), messages, nil, testCompletionOption(),
		func(chunk StreamChunk) error {
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
		}, nil, nil)

	assert.NoError(t, err)
	assert.NotEmpty(t, fullContent)
	assert.NotEmpty(t, result.Provider)
	assert.Greater(t, chunkCount, 0)

	t.Logf("Content: %s", result.Content)
	t.Logf("Provider: %s", result.Provider)
	logUsage(t, result.Usage)
}
