//go:build usage

package aiwire

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

// Anthropic requires >=1024 tokens to cache; the nonce forces a cold cache per run.
func buildLongSystemPrompt() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Session id: %d\n", time.Now().UnixNano())
	b.WriteString("You are a meticulous assistant. Follow these instructions carefully.\n\n")
	for i := 0; i < 200; i++ {
		b.WriteString("Rule: be concise, accurate, and never invent facts. Cite sources when possible. ")
		b.WriteString("Prefer short answers. Avoid filler. Respect user's time. Use simple English. ")
		b.WriteString("If unsure, say so. Do not speculate beyond what is asked. ")
	}
	return b.String()
}

func systemMessageWithCacheControl(text string) openai.ChatCompletionMessageParamUnion {
	part := openai.ChatCompletionContentPartTextParam{Text: text}
	part.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})
	return openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{part})
}

func runUsageCacheTest(t *testing.T, model string) {
	t.Helper()
	runUsageCacheTestWithProvider(t, model, &ProviderOption{AllowFallbacks: true}, false)
}

func runUsageCacheTestWithProvider(t *testing.T, model string, provider *ProviderOption, expectCacheWrite bool) {
	t.Helper()
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		systemMessageWithCacheControl(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	opts := CompletionOption{
		Model:       model,
		Temperature: 0.0,
		Provider:    provider,
	}

	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[%s] first call provider=%s", model, first.Provider)
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[%s] second call provider=%s", model, second.Provider)
	logUsage(t, second.Usage)

	if expectCacheWrite {
		assert.Greater(t, first.Usage.PromptTokensDetails.CacheCreationTokens, int64(0),
			"expected cache_write tokens on first call for %s", model)
	}
	assert.Greater(t, second.Usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second call for %s", model)
}

func TestUsage_OpenRouter_AnthropicSonnet46(t *testing.T) {
	runUsageCacheTestWithProvider(t, "anthropic/claude-sonnet-4.6", &ProviderOption{
		Order:          []string{"anthropic"},
		AllowFallbacks: false,
	}, true)
}

func TestUsage_OpenRouter_KimiK25(t *testing.T) {
	runUsageCacheTest(t, "moonshotai/kimi-k2.5")
}

func TestUsage_OpenAI_Direct(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.openai.com/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	opts := CompletionOption{
		Model:       "gpt-5.4-mini",
		Temperature: 0.0,
	}

	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[openai/gpt-4.1-mini] first call")
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[openai/gpt-4.1-mini] second call")
	logUsage(t, second.Usage)

	assert.Greater(t, second.Usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second OpenAI call")
	assert.Equal(t, int64(0), first.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, int64(0), second.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, float64(0), first.Usage.Cost, "OpenAI direct does not return cost")
}
