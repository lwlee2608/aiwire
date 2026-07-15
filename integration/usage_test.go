//go:build integration

package integration

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

const (
	openrouterURL = "https://openrouter.ai/api/v1"
	openaiURL     = "https://api.openai.com/v1"
	openrouterKey = "OPENROUTER_API_KEY"
	openaiKey     = "OPENAI_API_KEY"
)

var (
	anthropicOnly = &aiwire.ProviderOption{Order: []string{"anthropic"}, AllowFallbacks: false}
	openaiOnly    = &aiwire.ProviderOption{Order: []string{"openai"}, AllowFallbacks: false}
	sortByLatency = &aiwire.ProviderOption{Sort: "latency", AllowFallbacks: true}
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

type streamResult struct {
	content  string
	provider string
	usage    *aiwire.Usage
}

func collectStream(t *testing.T, service *aiwire.Service, ctx context.Context, params openai.ChatCompletionNewParams, provider *aiwire.ProviderOption) (streamResult, error) {
	t.Helper()
	var result streamResult
	err := service.ParamsCompletionsStream(ctx, params, provider, nil, func(chunk aiwire.StreamChunk) error {
		if chunk.Provider != "" {
			result.provider = chunk.Provider
		}
		if chunk.Done {
			result.usage = chunk.Usage
			return nil
		}
		result.content += chunk.Content
		return nil
	})
	return result, err
}

type usageCase struct {
	baseURL          string
	apiKeyEnv        string
	model            string
	provider         *aiwire.ProviderOption
	useCacheControl  bool
	expectCacheWrite bool // first call must report cache_write > 0
	forbidCacheWrite bool // both calls must report cache_write == 0
	expectZeroCost   bool // first.Cost must be 0 (OpenAI direct)
}

func (c usageCase) messages() []openai.ChatCompletionMessageParamUnion {
	system := buildLongSystemPrompt()
	sys := openai.SystemMessage(system)
	if c.useCacheControl {
		sys = systemMessageWithCacheControl(system)
	}
	return []openai.ChatCompletionMessageParamUnion{
		sys,
		openai.UserMessage("Reply with the single word: hello."),
	}
}

func (c usageCase) checkUsage(t *testing.T, first, second aiwire.Usage) {
	t.Helper()
	if c.expectCacheWrite {
		assert.Greater(t, first.PromptTokensDetails.CacheCreationTokens, int64(0),
			"expected cache_write tokens on first call for %s", c.model)
	}
	if c.forbidCacheWrite {
		assert.Equal(t, int64(0), first.PromptTokensDetails.CacheCreationTokens,
			"unexpected cache_write on first call for %s", c.model)
		assert.Equal(t, int64(0), second.PromptTokensDetails.CacheCreationTokens,
			"unexpected cache_write on second call for %s", c.model)
	}
	assert.Greater(t, second.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second call for %s", c.model)
	if c.expectZeroCost {
		assert.Equal(t, float64(0), first.Cost, "expected zero cost for %s", c.model)
	}
}

func (c usageCase) run(t *testing.T) {
	t.Helper()
	service := aiwire.NewOpenAIService(keyOrSkip(t, c.apiKeyEnv), c.baseURL)
	messages := c.messages()
	opts := aiwire.CompletionOption{Model: c.model, Temperature: 0.0, Provider: c.provider}
	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[%s] first call provider=%s", c.model, first.Provider)
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[%s] second call provider=%s", c.model, second.Provider)
	logUsage(t, second.Usage)

	c.checkUsage(t, first.Usage, second.Usage)
}

func (c usageCase) runStream(t *testing.T) {
	t.Helper()
	service := aiwire.NewOpenAIService(keyOrSkip(t, c.apiKeyEnv), c.baseURL)
	params := openai.ChatCompletionNewParams{
		Messages:    c.messages(),
		Model:       c.model,
		Temperature: 0.0,
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}
	ctx := context.Background()

	first, err := collectStream(t, service, ctx, params, c.provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.content)
	t.Logf("[%s stream] first call provider=%s", c.model, first.provider)
	if !assert.NotNil(t, first.usage, "expected usage on first stream call for %s", c.model) {
		return
	}
	logUsage(t, *first.usage)

	second, err := collectStream(t, service, ctx, params, c.provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.content)
	t.Logf("[%s stream] second call provider=%s", c.model, second.provider)
	if !assert.NotNil(t, second.usage, "expected usage on second stream call for %s", c.model) {
		return
	}
	logUsage(t, *second.usage)

	c.checkUsage(t, *first.usage, *second.usage)
}

func openrouterCase(model string, provider *aiwire.ProviderOption) usageCase {
	return usageCase{
		baseURL:         openrouterURL,
		apiKeyEnv:       openrouterKey,
		model:           model,
		provider:        provider,
		useCacheControl: true,
	}
}

func TestUsage_OpenRouter_AnthropicSonnet46(t *testing.T) {
	c := openrouterCase("anthropic/claude-sonnet-4.6", anthropicOnly)
	c.expectCacheWrite = true
	c.run(t)
}

func TestUsage_OpenRouter_AnthropicSonnet46_Stream(t *testing.T) {
	c := openrouterCase("anthropic/claude-sonnet-4.6", anthropicOnly)
	c.expectCacheWrite = true
	c.runStream(t)
}

func TestUsage_OpenRouter_KimiK25(t *testing.T) {
	openrouterCase("moonshotai/kimi-k2.5", sortByLatency).run(t)
}

func TestUsage_OpenRouter_KimiK25_Stream(t *testing.T) {
	openrouterCase("moonshotai/kimi-k2.5", sortByLatency).runStream(t)
}

func TestUsage_OpenRouter_GLM47(t *testing.T) {
	openrouterCase("z-ai/glm-4.7", sortByLatency).run(t)
}

func TestUsage_OpenRouter_GLM47_Stream(t *testing.T) {
	openrouterCase("z-ai/glm-4.7", sortByLatency).runStream(t)
}

func TestUsage_OpenRouter_OpenAIGPT5Mini(t *testing.T) {
	c := openrouterCase("openai/gpt-5-mini", openaiOnly)
	c.useCacheControl = false
	c.forbidCacheWrite = true
	c.run(t)
}

func TestUsage_OpenRouter_OpenAIGPT5Mini_Stream(t *testing.T) {
	c := openrouterCase("openai/gpt-5-mini", openaiOnly)
	c.useCacheControl = false
	c.forbidCacheWrite = true
	c.runStream(t)
}

func openaiDirectCase() usageCase {
	return usageCase{
		baseURL:          openaiURL,
		apiKeyEnv:        openaiKey,
		model:            "gpt-5.4-mini",
		forbidCacheWrite: true,
		expectZeroCost:   true,
	}
}

func TestUsage_OpenAI_Direct(t *testing.T) {
	openaiDirectCase().run(t)
}

func TestUsage_OpenAI_Direct_Stream(t *testing.T) {
	openaiDirectCase().runStream(t)
}
