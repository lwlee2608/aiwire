// Package aiwire provides a lightweight framework for building AI agents with
// tool-calling support. It works with any OpenAI-compatible API and offers both
// streaming and non-streaming completions, an agentic loop with automatic tool
// execution, and token usage tracking across multi-step interactions.
package aiwire

import (
	"context"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/responses"
)

type CompletionOption struct {
	Model          string
	Temperature    float64
	Provider       *ProviderOption
	MaxTokens      *int
	ResponseFormat openai.ChatCompletionNewParamsResponseFormatUnion
	Reasoning      *ReasoningOption
}

type ProviderDataCollection string

const (
	ProviderDataCollectionAllow ProviderDataCollection = "allow"
	ProviderDataCollectionDeny  ProviderDataCollection = "deny"
)

type ProviderOption struct {
	AllowFallbacks    bool
	RequireParameters bool
	DataCollection    ProviderDataCollection
	Sort              string
	Order             []string
	Ignore            []string
}

type Usage struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64

	PromptTokensDetails     PromptTokensDetails
	CompletionTokensDetails CompletionTokensDetails

	CacheDiscount float64
	Cost          float64
}

type PromptTokensDetails struct {
	AudioTokens         int64
	CachedTokens        int64
	CacheCreationTokens int64
}

type CompletionTokensDetails struct {
	AcceptedPredictionTokens int64
	AudioTokens              int64
	ReasoningTokens          int64
	RejectedPredictionTokens int64
}

type CompletionResponse struct {
	Message          openai.ChatCompletionMessage
	Reasoning        string
	ReasoningDetails []ReasoningDetail
	Provider         string
	Usage            Usage
}

type StreamChunk struct {
	Content          string
	Reasoning        string
	ReasoningDetails []ReasoningDetail
	Provider         string
	Role             string
	FinishReason     string
	Done             bool
	Usage            *Usage
	ToolCalls        []openai.ChatCompletionMessageToolCallUnion
	ToolResults      []StreamToolResult
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

type ResponsesOption struct {
	Model           string
	Temperature     float64
	MaxOutputTokens *int
	Instructions    string
	// Honored by OpenAI; OpenRouter ignores it (Beta is stateless).
	PreviousResponseID string
	Store              *bool
	Provider           *ProviderOption
	Reasoning          *ReasoningOption
	Include            []string
	ResponseFormat     responses.ResponseFormatTextConfigUnionParam
}

type ResponsesResponse struct {
	ID       string
	Status   string
	Output   []responses.ResponseOutputItemUnion
	Provider string
	Usage    Usage
}

type ResponsesStreamChunk struct {
	Type         string
	Delta        string
	Item         *responses.ResponseOutputItemUnion
	ResponseID   string
	Provider     string
	FinishReason string
	Done         bool
	Usage        *Usage
}

type ResponsesStreamCallback func(chunk ResponsesStreamChunk) error

type Responses interface {
	Respond(
		ctx context.Context,
		input responses.ResponseInputParam,
		tools []responses.ToolUnionParam,
		opt ResponsesOption,
	) (ResponsesResponse, error)

	RespondStream(
		ctx context.Context,
		input responses.ResponseInputParam,
		tools []responses.ToolUnionParam,
		opt ResponsesOption,
		callback ResponsesStreamCallback,
	) error
}

type Embedding interface {
	Embedding(ctx context.Context, input string, model string) ([]float32, error)
}
