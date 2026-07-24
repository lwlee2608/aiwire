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

// CompletionOption configures a chat completion request.
type CompletionOption struct {
	Model           string
	Temperature     float64
	OmitTemperature bool
	Provider        *ProviderOption
	MaxTokens       *int
	ResponseFormat  openai.ChatCompletionNewParamsResponseFormatUnion
	Reasoning       *ReasoningOption
}

// ProviderDataCollection controls whether the upstream provider may retain request data.
type ProviderDataCollection string

const (
	ProviderDataCollectionAllow ProviderDataCollection = "allow"
	ProviderDataCollectionDeny  ProviderDataCollection = "deny"
)

// ProviderOption holds OpenRouter-style provider routing preferences.
// Forwarded as the `provider` field on the request body.
type ProviderOption struct {
	AllowFallbacks    bool
	RequireParameters bool
	DataCollection    ProviderDataCollection
	Sort              string
	Order             []string
	Ignore            []string
}

// Usage is the normalized token accounting returned by a completion or response.
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

// StreamChunk is one event in a streaming completion. Done marks the
// terminal chunk, at which point Usage and ReasoningDetails (if any) are populated.
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

// StreamCallback receives each StreamChunk. Returning a non-nil error aborts the stream.
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

// Experimental: the Responses API surface (ResponsesOption, ResponsesResponse,
// ResponsesStreamChunk, ResponsesStreamCallback, and the Responses interface)
// is in beta and may change without notice. OpenRouter's /v1/responses is
// itself a beta endpoint and is stateless (PreviousResponseID / Store ignored).
type ResponsesOption struct {
	Model           string
	Temperature     float64
	OmitTemperature bool
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

// Experimental: see [ResponsesOption].
type ResponsesResponse struct {
	ID       string
	Status   string
	Output   []responses.ResponseOutputItemUnion
	Provider string
	Usage    Usage
}

// Experimental: see [ResponsesOption].
type ResponsesStreamChunk struct {
	Type        string
	Delta       string
	Item        *responses.ResponseOutputItemUnion
	ItemID      string
	OutputIndex int64
	ResponseID  string
	Provider    string
	Status      string
	Done        bool
	Usage       *Usage
}

// Experimental: see [ResponsesOption].
type ResponsesStreamCallback func(chunk ResponsesStreamChunk) error

// Experimental: see [ResponsesOption].
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
	EmbeddingBatch(ctx context.Context, inputs []string, model string) ([][]float32, error)
}
