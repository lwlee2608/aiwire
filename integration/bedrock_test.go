//go:build integration

package integration

import (
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
)

const (
	bedrockGPTModel    = "openai.gpt-oss-120b-1:0"
	bedrockClaudeModel = "us.anthropic.claude-sonnet-5"
)

func bedrockRegion() string {
	if region := os.Getenv("AWS_REGION"); region != "" {
		return region
	}
	return "us-west-2"
}

func bedrockService(t *testing.T) *aiwire.Service {
	apiKey := keyOrSkip(t, "AWS_BEARER_TOKEN_BEDROCK")
	return aiwire.NewOpenAIService(apiKey, "https://bedrock-runtime."+bedrockRegion()+".amazonaws.com/openai/v1")
}

func bedrockClaudeService(t *testing.T) *aiwire.AnthropicService {
	apiKey := keyOrSkip(t, "AWS_BEARER_TOKEN_BEDROCK")
	return aiwire.NewAnthropicService(apiKey, bedrockRegion())
}

func TestBedrock_Completion(t *testing.T) {
	service := bedrockService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       bedrockGPTModel,
		Temperature: 0.7,
	})
}

func TestBedrock_Streaming(t *testing.T) {
	service := bedrockService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       bedrockGPTModel,
		Temperature: 0.7,
	})
}

func TestBedrock_ClaudeCompletion(t *testing.T) {
	service := bedrockClaudeService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You are a comedian."),
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       bedrockClaudeModel,
		Temperature: 0.7,
	})
}

func TestBedrock_ClaudeStreaming(t *testing.T) {
	t.Skip("AnthropicService streaming is not implemented")

	service := bedrockClaudeService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       bedrockClaudeModel,
		Temperature: 0.7,
	})
}
