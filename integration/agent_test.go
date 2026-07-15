//go:build integration

package integration

import (
	"context"
	"fmt"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)

type addInput struct {
	A int `json:"a" jsonschema:"required"`
	B int `json:"b" jsonschema:"required"`
}

type addTool struct{}

var _ aiwire.Tool = (*addTool)(nil)

func (t *addTool) Definition() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "add",
		Description: openai.String("Add two integers"),
		Parameters:  aiwire.GenerateFunctionParameters[addInput](),
	})
}

func (t *addTool) Execute(ctx context.Context, inputs map[string]any) (aiwire.ToolResult, error) {
	a := inputs["a"].(float64)
	b := inputs["b"].(float64)
	return &aiwire.SimpleToolResult{ToolContent: fmt.Sprintf("%d", int(a)+int(b))}, nil
}

func newTestAgent(t *testing.T) *aiwire.Agent {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	return aiwire.NewAgent(service, 5)
}

func testCompletionOption() aiwire.CompletionOption {
	return aiwire.CompletionOption{
		Model:       "moonshotai/kimi-k2.5",
		Temperature: 0.7,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &aiwire.ReasoningOption{
			Effort: aiwire.ReasoningEffortLow,
		},
	}
}

func TestAgent_Execute(t *testing.T) {
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

func TestAgent_Execute_ToolCall(t *testing.T) {
	agent := newTestAgent(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You must use the add tool to answer any arithmetic question."),
		openai.UserMessage("What is 17 + 28?"),
	}

	var preCalls []aiwire.PreTool
	var postCalls []aiwire.PostTool
	result, err := agent.Execute(context.Background(), messages, []aiwire.Tool{&addTool{}},
		testCompletionOption(),
		func(p aiwire.PreTool) { preCalls = append(preCalls, p) },
		func(p aiwire.PostTool) { postCalls = append(postCalls, p) },
	)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 1)
	assert.Equal(t, "add", result.ToolCalls[0].Function.Name)
	assert.Equal(t, len(result.ToolCalls), len(result.ToolResults))
	assert.Equal(t, "45", result.ToolResults[0].Content())
	assert.Contains(t, result.Content, "45")
	assert.Equal(t, len(result.ToolCalls), len(preCalls))
	assert.Equal(t, len(result.ToolCalls), len(postCalls))

	t.Logf("Content: %s", result.Content)
	t.Logf("Provider: %s", result.Provider)
	t.Logf("Tool call args: %s", result.ToolCalls[0].Function.Arguments)
	logUsage(t, result.Usage)
}

func TestAgent_ExecuteStream(t *testing.T) {
	agent := newTestAgent(t)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	var fullContent string
	var chunkCount int

	result, err := agent.ExecuteStream(context.Background(), messages, nil, testCompletionOption(),
		func(chunk aiwire.StreamChunk) error {
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

// effort:high + chained tool calls reliably produces encrypted reasoning on
// every step; without replay the follow-up call drops it and (sometimes)
// errors. NoError + reasoning_tokens>0 proves round-trip works.
func TestAgent_Execute_ReasoningReplay_GPT55(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	agent := aiwire.NewAgent(service, 5)

	option := aiwire.CompletionOption{
		Model:       "openai/gpt-5.5",
		Temperature: 0.0,
		Provider:    &aiwire.ProviderOption{Order: []string{"openai"}, AllowFallbacks: false},
		Reasoning: &aiwire.ReasoningOption{
			Effort:  aiwire.ReasoningEffortHigh,
			Summary: "detailed",
		},
	}

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You can only add two numbers at a time using the add tool. To compute longer sums, call add multiple times."),
		openai.UserMessage("Use the add tool to compute the sum 11 + 22 + 33 + 44. Show your work with separate add calls and report only the final number."),
	}

	result, err := agent.Execute(context.Background(), messages, []aiwire.Tool{&addTool{}}, option, nil, nil)

	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(result.ToolCalls), 2, "expected multiple tool calls (chained add)")
	assert.Contains(t, result.Content, "110")
	assert.Greater(t, result.Usage.CompletionTokensDetails.ReasoningTokens, int64(0),
		"expected reasoning tokens to be produced over multi-step run")

	t.Logf("Content: %s", result.Content)
	t.Logf("Provider: %s", result.Provider)
	logUsage(t, result.Usage)
}
