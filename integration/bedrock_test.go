//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	bedrockGPTModel    = "openai.gpt-oss-120b-1:0"
	bedrockClaudeModel = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
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
	service := bedrockClaudeService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       bedrockClaudeModel,
		Temperature: 0.7,
	})
}

func TestBedrock_ClaudeToolCall(t *testing.T) {
	agent := aiwire.NewAgent(bedrockClaudeService(t), 5)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You must use the add tool to answer any arithmetic question."),
		openai.UserMessage("What is 17 + 28?"),
	}

	result, err := agent.Execute(context.Background(), messages, []aiwire.Tool{&addTool{}},
		aiwire.CompletionOption{Model: bedrockClaudeModel, Temperature: 0.0}, nil, nil)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 1)
	assert.Equal(t, "add", result.ToolCalls[0].Function.Name)
	assert.Equal(t, "45", result.ToolResults[0].Content())
	assert.Contains(t, result.Content, "45")

	t.Logf("Content: %s", result.Content)
	logUsage(t, result.Usage)
}

func TestBedrock_ClaudeToolCallStreaming(t *testing.T) {
	agent := aiwire.NewAgent(bedrockClaudeService(t), 5)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You must use the add tool to answer any arithmetic question."),
		openai.UserMessage("What is 17 + 28?"),
	}

	var streamed string
	result, err := agent.ExecuteStream(context.Background(), messages, []aiwire.Tool{&addTool{}},
		aiwire.CompletionOption{Model: bedrockClaudeModel, Temperature: 0.0},
		func(chunk aiwire.StreamChunk) error {
			streamed += chunk.Content
			return nil
		}, nil, nil)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 1)
	assert.Equal(t, "add", result.ToolCalls[0].Function.Name)
	assert.Equal(t, "45", result.ToolResults[0].Content())
	assert.Contains(t, streamed, "45")

	t.Logf("Streamed: %s", streamed)
	logUsage(t, result.Usage)
}

func TestBedrock_ClaudeThinking(t *testing.T) {
	agent := aiwire.NewAgent(bedrockClaudeService(t), 5)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You can only add two numbers at a time using the add tool. To compute longer sums, call add multiple times."),
		openai.UserMessage("Use the add tool to compute 11 + 22 + 33 + 44. Report only the final number."),
	}

	budget := 1024
	maxTokens := 4096
	result, err := agent.Execute(context.Background(), messages, []aiwire.Tool{&addTool{}},
		aiwire.CompletionOption{
			Model:           bedrockClaudeModel,
			OmitTemperature: true,
			MaxTokens:       &maxTokens,
			Reasoning:       &aiwire.ReasoningOption{MaxTokens: &budget},
		}, nil, nil)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 2, "expected chained add calls")
	assert.Contains(t, result.Content, "110")

	t.Logf("Content: %s", result.Content)
	logUsage(t, result.Usage)
}

func TestBedrock_ClaudeResponseFormat(t *testing.T) {
	service := bedrockClaudeService(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("What is the capital of France and its population?"),
	}

	resp, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:           bedrockClaudeModel,
		OmitTemperature: true,
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "capital",
					Strict: openai.Bool(true),
					Schema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city":       map[string]any{"type": "string"},
							"population": map[string]any{"type": "integer"},
						},
						"required":             []string{"city", "population"},
						"additionalProperties": false,
					},
				},
			},
		},
	})
	require.NoError(t, err)

	var out struct {
		City       string `json:"city"`
		Population int64  `json:"population"`
	}
	require.NoError(t, json.Unmarshal([]byte(resp.Message.Content), &out))
	assert.Equal(t, "Paris", out.City)
	assert.Greater(t, out.Population, int64(0))

	t.Logf("Content: %s", resp.Message.Content)
}
