//go:build openrouter

package aiwire

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)

type addInput struct {
	A float64 `json:"a" jsonschema:"required"`
	B float64 `json:"b" jsonschema:"required"`
}

type addTool struct{}

func (t *addTool) Definition() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "add",
		Description: openai.String("Add two integers"),
		Parameters:  GenerateFunctionParameters[addInput](),
	})
}

func (t *addTool) Execute(ctx context.Context, inputs map[string]any) (ToolResult, error) {
	a := inputs["a"].(float64)
	b := inputs["b"].(float64)
	return &SimpleToolResult{ToolContent: fmt.Sprintf("%d", int(a+b))}, nil
}

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

func TestAgent_Execute_ToolCall_OpenRouter(t *testing.T) {
	agent := newTestAgent(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You must use the add tool to answer any arithmetic question."),
		openai.UserMessage("What is 2 + 3?"),
	}

	var preCalls []PreTool
	var postCalls []PostTool
	result, err := agent.Execute(context.Background(), messages, []Tool{&addTool{}},
		testCompletionOption(),
		func(p PreTool) { preCalls = append(preCalls, p) },
		func(p PostTool) { postCalls = append(postCalls, p) },
	)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 1)
	assert.Equal(t, "add", result.ToolCalls[0].Function.Name)
	assert.Equal(t, len(result.ToolCalls), len(result.ToolResults))
	assert.Equal(t, "5", result.ToolResults[0].Content())
	assert.Contains(t, result.Content, "5")
	assert.Equal(t, len(result.ToolCalls), len(preCalls))
	assert.Equal(t, len(result.ToolCalls), len(postCalls))

	t.Logf("Content: %s", result.Content)
	t.Logf("Provider: %s", result.Provider)
	t.Logf("Tool call args: %s", result.ToolCalls[0].Function.Arguments)
	logUsage(t, result.Usage)
}

func TestAgent_ExecuteStream_OpenRouter(t *testing.T) {
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
