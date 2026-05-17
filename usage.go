package aiwire

import (
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/respjson"
	"github.com/openai/openai-go/v3/responses"
)

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

func UsageFromResponses(u responses.ResponseUsage) Usage {
	out := Usage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
		PromptTokensDetails: PromptTokensDetails{
			CachedTokens: u.InputTokensDetails.CachedTokens,
		},
		CompletionTokensDetails: CompletionTokensDetails{
			ReasoningTokens: u.OutputTokensDetails.ReasoningTokens,
		},
	}

	if v, ok := extraInt64(u.InputTokensDetails.JSON.ExtraFields, "cache_creation_tokens"); ok {
		out.PromptTokensDetails.CacheCreationTokens = v
	} else if v, ok := extraInt64(u.InputTokensDetails.JSON.ExtraFields, "cache_write_tokens"); ok {
		out.PromptTokensDetails.CacheCreationTokens = v
	}
	if v, ok := extraInt64(u.JSON.ExtraFields, "cache_creation_input_tokens"); ok {
		out.PromptTokensDetails.CacheCreationTokens = v
	}

	if v, ok := extraFloat64(u.JSON.ExtraFields, "cache_discount"); ok {
		out.CacheDiscount = v
	}
	if v, ok := extraFloat64(u.JSON.ExtraFields, "cost"); ok {
		out.Cost = v
	}
	return out
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
