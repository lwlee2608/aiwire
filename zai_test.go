//go:build zai

package aiwire

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestZAI_Completion(t *testing.T) {
	apiKey := os.Getenv("ZAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}

func TestZAI_Streaming(t *testing.T) {
	apiKey := os.Getenv("ZAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}
