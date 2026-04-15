//go:build openrouter

package aiwire

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)


func TestOpenAI_Completion_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "z-ai/glm-4.7",
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
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "z-ai/glm-4.7",
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
