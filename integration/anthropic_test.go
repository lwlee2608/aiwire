//go:build integration

package integration

import (
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
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
