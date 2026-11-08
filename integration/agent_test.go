//go:build integration

package integration

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

type slowLookupTool struct {
	name  string
	value string
}

var _ aiwire.ConcurrentTool = (*slowLookupTool)(nil)

func (t *slowLookupTool) Definition() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        t.name,
		Description: openai.String("Look up the current " + strings.TrimPrefix(t.name, "get_") + " for a city"),
		Parameters:  aiwire.GenerateFunctionParameters[cityInput](),
	})
}

func (t *slowLookupTool) Execute(ctx context.Context, _ map[string]any) (aiwire.ToolResult, error) {
	select {
	case <-time.After(time.Second):
		return &aiwire.SimpleToolResult{ToolContent: t.value}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (t *slowLookupTool) Concurrent() bool { return true }

type cityInput struct {
	City string `json:"city" jsonschema:"required"`
}

func TestAgent_Execute_ParallelTools(t *testing.T) {
	gptOption := aiwire.CompletionOption{
		Model:           "openai/gpt-6.1-terra",
		OmitTemperature: true,
		Reasoning:       &aiwire.ReasoningOption{Effort: aiwire.ReasoningEffortLow},
	}
	for _, option := range []aiwire.CompletionOption{testCompletionOption(), gptOption} {
		t.Run(option.Model, func(t *testing.T) {
			runParallelToolsTest(t, option)
		})
	}
}

func runParallelToolsTest(t *testing.T, option aiwire.CompletionOption) {
	agent := newTestAgent(t)
	tools := []aiwire.Tool{
		&slowLookupTool{name: "get_weather", value: "sunny, 24C"},
		&slowLookupTool{name: "get_time", value: "14:05"},
		&slowLookupTool{name: "get_population", value: "5.9 million"},
	}
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("Call every tool you need in a single response, in parallel."),
		openai.UserMessage("For Singapore, what is the weather, the local time, and the population? Call get_weather, get_time and get_population at once."),
	}

	var mu sync.Mutex
	var events []string
	var firstPre, lastPost time.Time
	result, err := agent.Execute(context.Background(), messages, tools, option,
		func(p aiwire.PreTool) {
			mu.Lock()
			defer mu.Unlock()
			if firstPre.IsZero() {
				firstPre = time.Now()
			}
			events = append(events, "pre:"+p.ID)
		},
		func(p aiwire.PostTool) {
			mu.Lock()
			defer mu.Unlock()
			lastPost = time.Now()
			events = append(events, "post:"+p.ID)
			t.Logf("%s (%s) took %s", p.Name, p.ID, p.Duration)
		},
	)

	require.NoError(t, err)
	assert.Len(t, result.ToolCalls, 3)
	require.Len(t, events, 6)
	for _, e := range events[:3] {
		assert.True(t, strings.HasPrefix(e, "pre:"), "expected all three calls in one step, got %v", events)
	}
	toolPhase := lastPost.Sub(firstPre)
	assert.Less(t, toolPhase, 2*time.Second, "step should take about the slowest tool (1s), not the sum (3s)")

	t.Logf("Tool phase: %s", toolPhase)
	t.Logf("Content: %s", result.Content)
	logUsage(t, result.Usage)
}
