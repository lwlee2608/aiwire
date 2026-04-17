package aiwire

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)

func schemaFor[T any]() shared.FunctionParameters {
	raw, _ := json.Marshal(GenerateSchema[T]())
	var params shared.FunctionParameters
	_ = json.Unmarshal(raw, &params)
	return params
}

type mockCompletion struct {
	responses []CompletionResponse
	calls     int
}

func (m *mockCompletion) Completions(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
) (CompletionResponse, error) {
	if m.calls >= len(m.responses) {
		return CompletionResponse{}, fmt.Errorf("unexpected call %d", m.calls)
	}
	resp := m.responses[m.calls]
	m.calls++
	return resp, nil
}

func (m *mockCompletion) CompletionsStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
	callback StreamCallback,
) error {
	return fmt.Errorf("not implemented")
}

func (m *mockCompletion) Models(ctx context.Context) (*pagination.Page[openai.Model], error) {
	return nil, fmt.Errorf("not implemented")
}

type addInput struct {
	A float64 `json:"a" jsonschema:"required"`
	B float64 `json:"b" jsonschema:"required"`
}

type addTool struct{}

func (t *addTool) Definition() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "add",
		Description: openai.String("Add two integers"),
		Parameters:  schemaFor[addInput](),
	})
}

func (t *addTool) Execute(ctx context.Context, inputs map[string]any) (ToolResult, error) {
	a := inputs["a"].(float64)
	b := inputs["b"].(float64)
	return &SimpleToolResult{ToolContent: fmt.Sprintf("%d", int(a+b))}, nil
}

func TestAgent_Execute_ToolCall(t *testing.T) {
	mock := &mockCompletion{
		responses: []CompletionResponse{
			{
				Message: openai.ChatCompletionMessage{
					Role: "assistant",
					ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
						{
							ID:   "call_1",
							Type: "function",
							Function: openai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "add",
								Arguments: `{"a": 2, "b": 3}`,
							},
						},
					},
				},
				Usage: openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			},
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "The answer is 5.",
				},
				Usage: openai.CompletionUsage{PromptTokens: 20, CompletionTokens: 6, TotalTokens: 26},
			},
		},
	}

	agent := NewAgent(mock, 5)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("What is 2 + 3?"),
	}

	var preCalls []PreTool
	var postCalls []PostTool
	result, err := agent.Execute(context.Background(), messages, []Tool{&addTool{}},
		CompletionOption{Model: "test"},
		func(p PreTool) { preCalls = append(preCalls, p) },
		func(p PostTool) { postCalls = append(postCalls, p) },
	)

	assert.NoError(t, err)
	assert.Equal(t, 2, mock.calls)
	assert.Equal(t, "The answer is 5.", result.Content)
	assert.Len(t, result.ToolCalls, 1)
	assert.Equal(t, "add", result.ToolCalls[0].Function.Name)
	assert.Len(t, result.ToolResults, 1)
	assert.Equal(t, "5", result.ToolResults[0].Content())
	assert.NoError(t, result.ToolResults[0].Error())

	assert.Len(t, preCalls, 1)
	assert.Equal(t, "add", preCalls[0].Name)
	assert.Equal(t, `{"a": 2, "b": 3}`, preCalls[0].Args)
	assert.Len(t, postCalls, 1)
	assert.Equal(t, "5", postCalls[0].Result.Content())

	assert.Equal(t, int64(30), result.Usage.PromptTokens)
	assert.Equal(t, int64(11), result.Usage.CompletionTokens)
	assert.Equal(t, int64(41), result.Usage.TotalTokens)
}
