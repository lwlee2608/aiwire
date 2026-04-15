//go:build openai

package aiwire

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_Completion(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: 0.7,
	})
}

func TestOpenAI_Streaming(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: 0.7,
	})
}
