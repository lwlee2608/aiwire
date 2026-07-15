//go:build integration

package integration

import (
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
)

func TestZAI_Completion(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "ZAI_API_KEY"), "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}

func TestZAI_Streaming(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "ZAI_API_KEY"), "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}
