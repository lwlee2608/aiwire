// Package aiwire provides a lightweight framework for building AI agents with
// tool-calling support. It works with any OpenAI-compatible API and offers both
// streaming and non-streaming completions, an agentic loop with automatic tool
// execution, and token usage tracking across multi-step interactions.
package aiwire

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/packages/respjson"
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
	Summary   string // OpenAI gpt-5 reasoning summary verbosity: "auto" | "concise" | "detailed"
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
	Message   openai.ChatCompletionMessage
	Reasoning string
	Provider  string
	Usage     Usage
}

type StreamChunk struct {
	Content      string
	Reasoning    string
	Provider     string
	Role         string
	FinishReason string
	Done         bool
	Usage        *Usage
	ToolCalls    []openai.ChatCompletionMessageToolCallUnion
	ToolResults  []StreamToolResult
}

func UsageFromOpenAI(u openai.CompletionUsage) Usage {
	out := Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		PromptTokensDetails: PromptTokensDetails{
			AudioTokens:  u.PromptTokensDetails.AudioTokens,
			CachedTokens: u.PromptTokensDetails.CachedTokens,
		},
		CompletionTokensDetails: CompletionTokensDetails{
			AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens,
			AudioTokens:              u.CompletionTokensDetails.AudioTokens,
			ReasoningTokens:          u.CompletionTokensDetails.ReasoningTokens,
			RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens,
		},
	}

	cacheCreationSet := false
	if v, ok := extraInt64(u.JSON.ExtraFields, "cache_creation_input_tokens"); ok {
		out.PromptTokensDetails.CacheCreationTokens = v
		cacheCreationSet = true
	}
	if !u.PromptTokensDetails.JSON.CachedTokens.Valid() {
		if v, ok := extraInt64(u.JSON.ExtraFields, "cache_read_input_tokens"); ok {
			out.PromptTokensDetails.CachedTokens = v
		}
	}
	if !cacheCreationSet {
		if v, ok := extraInt64(u.PromptTokensDetails.JSON.ExtraFields, "cache_creation_tokens"); ok {
			out.PromptTokensDetails.CacheCreationTokens = v
		} else if v, ok := extraInt64(u.PromptTokensDetails.JSON.ExtraFields, "cache_write_tokens"); ok {
			out.PromptTokensDetails.CacheCreationTokens = v
		}
	}

	if v, ok := extraFloat64(u.JSON.ExtraFields, "cache_discount"); ok {
		out.CacheDiscount = v
	}
	if v, ok := extraFloat64(u.JSON.ExtraFields, "cost"); ok {
		out.Cost = v
	}
	return out
}

func extraInt64(extras map[string]respjson.Field, key string) (int64, bool) {
	f, ok := extras[key]
	if !ok {
		return 0, false
	}
	raw := f.Raw()
	if raw == "" || raw == "null" {
		return 0, false
	}
	var v int64
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return 0, false
	}
	return v, true
}

func extraFloat64(extras map[string]respjson.Field, key string) (float64, bool) {
	f, ok := extras[key]
	if !ok {
		return 0, false
	}
	raw := f.Raw()
	if raw == "" || raw == "null" {
		return 0, false
	}
	var v float64
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return 0, false
	}
	return v, true
}

func (u *Usage) Add(src Usage) {
	u.PromptTokens += src.PromptTokens
	u.CompletionTokens += src.CompletionTokens
	u.TotalTokens += src.TotalTokens

	u.PromptTokensDetails.AudioTokens += src.PromptTokensDetails.AudioTokens
	u.PromptTokensDetails.CachedTokens += src.PromptTokensDetails.CachedTokens
	u.PromptTokensDetails.CacheCreationTokens += src.PromptTokensDetails.CacheCreationTokens

	u.CompletionTokensDetails.AcceptedPredictionTokens += src.CompletionTokensDetails.AcceptedPredictionTokens
	u.CompletionTokensDetails.AudioTokens += src.CompletionTokensDetails.AudioTokens
	u.CompletionTokensDetails.ReasoningTokens += src.CompletionTokensDetails.ReasoningTokens
	u.CompletionTokensDetails.RejectedPredictionTokens += src.CompletionTokensDetails.RejectedPredictionTokens

	u.CacheDiscount += src.CacheDiscount
	u.Cost += src.Cost
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

type Embedding interface {
	Embedding(ctx context.Context, input string, model string) ([]float32, error)
}
