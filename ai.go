package ai

import (
	"context"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
)

type CompletionOption struct {
	Model          string
	Temperature    float64
	Provider       *ProviderOption
	MaxTokens      *int
	ResponseFormat openai.ChatCompletionNewParamsResponseFormatUnion
	Reasoning      *ReasoningOption
}

type ReasoningEffort string

const (
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
)

type ReasoningOption struct {
	Effort    ReasoningEffort
	MaxTokens *int
	Exclude   bool
}

type ProviderOption struct {
	AllowFallbacks bool
	Sort           string
}

type CompletionResponse struct {
	Message   openai.ChatCompletionMessage
	Reasoning string
	Provider  string
	Usage     openai.CompletionUsage
}

type StreamChunk struct {
	Content      string
	Reasoning    string
	Provider     string
	Role         string
	FinishReason string
	Done         bool
	Usage        *openai.CompletionUsage
	ToolCalls    []openai.ChatCompletionMessageToolCallUnion
	ToolResults  []StreamToolResult
}

type StreamToolResult struct {
	Name    string
	Content string
	Error   error
}

type StreamCallback func(chunk StreamChunk) error

type Completion interface {
	Completions(
		ctx context.Context,
		messages []openai.ChatCompletionMessageParamUnion,
		tools []openai.ChatCompletionToolUnionParam,
		option CompletionOption) (CompletionResponse, error)

	CompletionsStream(
		ctx context.Context,
		messages []openai.ChatCompletionMessageParamUnion,
		tools []openai.ChatCompletionToolUnionParam,
		option CompletionOption,
		callback StreamCallback) error

	Models(ctx context.Context) (*pagination.Page[openai.Model], error)
}
