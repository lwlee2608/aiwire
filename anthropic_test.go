package aiwire

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func blockJSON(t *testing.T, block anthropic.ContentBlockParamUnion) map[string]any {
	t.Helper()
	raw, err := block.MarshalJSON()
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal(raw, &out))
	return out
}

func TestAnthropicMessages_SystemAndTurns(t *testing.T) {
	system, messages, err := anthropicMessages([]openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("be terse"),
		openai.UserMessage("hello"),
		openai.AssistantMessage("hi"),
		openai.UserMessage("bye"),
	})

	require.NoError(t, err)
	require.Len(t, system, 1)
	assert.Equal(t, "be terse", system[0].Text)

	require.Len(t, messages, 3)
	assert.Equal(t, anthropic.MessageParamRoleUser, messages[0].Role)
	assert.Equal(t, anthropic.MessageParamRoleAssistant, messages[1].Role)
	assert.Equal(t, anthropic.MessageParamRoleUser, messages[2].Role)
	assert.Equal(t, "hello", blockJSON(t, messages[0].Content[0])["text"])
}

// Parallel tool results arrive as separate OpenAI messages but must reach
// Anthropic as one user turn.
func TestAnthropicMessages_MergesConsecutiveToolResults(t *testing.T) {
	_, messages, err := anthropicMessages([]openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("add some numbers"),
		AssistantMessageWithReasoning("", []openai.ChatCompletionMessageToolCallUnion{
			{ID: "call_1", Type: "function", Function: openai.ChatCompletionMessageFunctionToolCallFunction{Name: "add", Arguments: `{"a":1,"b":2}`}},
			{ID: "call_2", Type: "function", Function: openai.ChatCompletionMessageFunctionToolCallFunction{Name: "add", Arguments: `{"a":3,"b":4}`}},
		}, nil),
		openai.ToolMessage("3", "call_1"),
		openai.ToolMessage("7", "call_2"),
	})

	require.NoError(t, err)
	require.Len(t, messages, 3)

	assistant := messages[1]
	assert.Equal(t, anthropic.MessageParamRoleAssistant, assistant.Role)
	require.Len(t, assistant.Content, 2, "empty assistant text must not become a block")
	assert.Equal(t, "tool_use", blockJSON(t, assistant.Content[0])["type"])

	results := messages[2]
	assert.Equal(t, anthropic.MessageParamRoleUser, results.Role)
	require.Len(t, results.Content, 2, "both tool results belong to one user turn")
	assert.Equal(t, "call_1", blockJSON(t, results.Content[0])["tool_use_id"])
	assert.Equal(t, "call_2", blockJSON(t, results.Content[1])["tool_use_id"])
}

func TestAnthropicMessages_ThinkingBlocksLeadTheTurn(t *testing.T) {
	details := []ReasoningDetail{anthropicReasoningText(0, "let me think", "sig-abc")}

	_, messages, err := anthropicMessages([]openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("hello"),
		AssistantMessageWithReasoning("hi", nil, details),
	})

	require.NoError(t, err)
	require.Len(t, messages, 2)
	require.Len(t, messages[1].Content, 2)

	thinking := blockJSON(t, messages[1].Content[0])
	assert.Equal(t, "thinking", thinking["type"])
	assert.Equal(t, "let me think", thinking["thinking"])
	assert.Equal(t, "sig-abc", thinking["signature"])
	assert.Equal(t, "text", blockJSON(t, messages[1].Content[1])["type"])
}

// A signature-less detail comes from another gateway and would be rejected.
func TestAnthropicMessages_DropsUnsignedThinking(t *testing.T) {
	details := []ReasoningDetail{{Type: reasoningDetailTypeText, Raw: json.RawMessage(`{"type":"reasoning.text","text":"no sig"}`)}}

	_, messages, err := anthropicMessages([]openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("hello"),
		AssistantMessageWithReasoning("hi", nil, details),
	})

	require.NoError(t, err)
	require.Len(t, messages[1].Content, 1)
	assert.Equal(t, "text", blockJSON(t, messages[1].Content[0])["type"])
}

func TestAnthropicTextContent(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		text, err := anthropicTextContent(openai.UserMessage("plain").OfUser.Content)
		require.NoError(t, err)
		assert.Equal(t, "plain", text)
	})

	t.Run("text parts joined", func(t *testing.T) {
		msg := openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("one "),
			openai.TextContentPart("two"),
		})
		text, err := anthropicTextContent(msg.OfUser.Content)
		require.NoError(t, err)
		assert.Equal(t, "one two", text)
	})

	t.Run("non-text part rejected", func(t *testing.T) {
		msg := openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: "https://example.com/a.png"}),
		})
		_, err := anthropicTextContent(msg.OfUser.Content)
		assert.ErrorContains(t, err, "unsupported content part")
	})
}

func TestAnthropicTools_MapsSchema(t *testing.T) {
	tools, err := anthropicTools([]openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        "add",
			Description: openai.String("Add two integers"),
			Parameters: shared.FunctionParameters{
				"type":                 "object",
				"properties":           map[string]any{"a": map[string]any{"type": "integer"}},
				"required":             []string{"a"},
				"additionalProperties": false,
			},
		}),
	})

	require.NoError(t, err)
	require.Len(t, tools, 1)

	tool := tools[0].OfTool
	require.NotNil(t, tool)
	assert.Equal(t, "add", tool.Name)
	assert.Equal(t, "Add two integers", tool.Description.Value)
	assert.Equal(t, []string{"a"}, tool.InputSchema.Required)
	assert.NotNil(t, tool.InputSchema.Properties)
	assert.Equal(t, false, tool.InputSchema.ExtraFields["additionalProperties"],
		"unrecognized schema keys must survive as extra fields")
}

func TestAnthropicStringSlice(t *testing.T) {
	assert.Equal(t, []string{"a"}, anthropicStringSlice([]string{"a"}))
	assert.Equal(t, []string{"a", "b"}, anthropicStringSlice([]any{"a", "b", 3}))
	assert.Nil(t, anthropicStringSlice("nope"))
}

func TestAnthropicFinishReason(t *testing.T) {
	assert.Equal(t, "stop", anthropicFinishReason("end_turn"))
	assert.Equal(t, "stop", anthropicFinishReason("stop_sequence"))
	assert.Equal(t, "length", anthropicFinishReason("max_tokens"))
	assert.Equal(t, "tool_calls", anthropicFinishReason("tool_use"))
	assert.Equal(t, "something_new", anthropicFinishReason("something_new"))
}

// A no-argument tool streams no input_json_delta, which previously left the
// arguments empty and unparseable by the agent loop.
func TestAnthropicStreamAccum_NoArgToolYieldsEmptyObject(t *testing.T) {
	acc := anthropicStreamAccum{blocks: map[int64]*anthropicStreamBlock{}}
	acc.start(0, anthropic.ContentBlockStartEventContentBlockUnion{
		Type: "tool_use",
		ID:   "call_1",
		Name: "get_time",
	})

	calls := acc.toolCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, "get_time", calls[0].Function.Name)
	assert.Equal(t, "{}", calls[0].Function.Arguments)

	var args map[string]any
	assert.NoError(t, json.Unmarshal([]byte(calls[0].Function.Arguments), &args))
}

func TestAnthropicStreamAccum_AccumulatesToolArguments(t *testing.T) {
	acc := anthropicStreamAccum{blocks: map[int64]*anthropicStreamBlock{}}
	acc.start(0, anthropic.ContentBlockStartEventContentBlockUnion{Type: "tool_use", ID: "call_1", Name: "add"})
	acc.blocks[0].toolCall.Function.Arguments += `{"a":1,`
	acc.blocks[0].toolCall.Function.Arguments += `"b":2}`

	calls := acc.toolCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, `{"a":1,"b":2}`, calls[0].Function.Arguments)
}

func TestAnthropicStreamAccum_ReasoningDetails(t *testing.T) {
	acc := anthropicStreamAccum{blocks: map[int64]*anthropicStreamBlock{}}
	acc.start(0, anthropic.ContentBlockStartEventContentBlockUnion{Type: "thinking"})
	acc.blocks[0].thinking.WriteString("step one ")
	acc.blocks[0].thinking.WriteString("step two")
	acc.blocks[0].signature = "sig-1"
	acc.start(1, anthropic.ContentBlockStartEventContentBlockUnion{Type: "redacted_thinking", Data: "blob"})
	acc.start(2, anthropic.ContentBlockStartEventContentBlockUnion{Type: "text"})

	details := acc.reasoningDetails()
	require.Len(t, details, 2, "text blocks are not reasoning")

	var text map[string]any
	require.NoError(t, json.Unmarshal(details[0].Raw, &text))
	assert.Equal(t, reasoningDetailTypeText, text["type"])
	assert.Equal(t, "step one step two", text["text"])
	assert.Equal(t, "sig-1", text["signature"])

	var encrypted map[string]any
	require.NoError(t, json.Unmarshal(details[1].Raw, &encrypted))
	assert.Equal(t, reasoningDetailTypeEncrypted, encrypted["type"])
	assert.Equal(t, "blob", encrypted["data"])
}

func TestAnthropicParams_Temperature(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}

	params, err := anthropicParams(messages, nil, CompletionOption{Model: "m", Temperature: 0.7})
	require.NoError(t, err)
	assert.Equal(t, 0.7, params.Temperature.Value)
	assert.Equal(t, int64(anthropicDefaultMaxTokens), params.MaxTokens)

	params, err = anthropicParams(messages, nil, CompletionOption{Model: "m", OmitTemperature: true})
	require.NoError(t, err)
	assert.False(t, params.Temperature.Valid())
}

func TestAnthropicParams_ThinkingBudget(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}

	t.Run("raises the default max_tokens above the budget", func(t *testing.T) {
		budget := 8192
		params, err := anthropicParams(messages, nil, CompletionOption{
			Model:     "m",
			Reasoning: &ReasoningOption{MaxTokens: &budget},
		})
		require.NoError(t, err)
		require.NotNil(t, params.Thinking.OfEnabled)
		assert.Equal(t, int64(8192), params.Thinking.OfEnabled.BudgetTokens)
		assert.Greater(t, params.MaxTokens, int64(8192))
		assert.False(t, params.Temperature.Valid(), "thinking pins temperature to 1")
	})

	t.Run("leaves an explicit max_tokens alone", func(t *testing.T) {
		budget := 1024
		maxTokens := 2048
		params, err := anthropicParams(messages, nil, CompletionOption{
			Model:     "m",
			MaxTokens: &maxTokens,
			Reasoning: &ReasoningOption{MaxTokens: &budget},
		})
		require.NoError(t, err)
		assert.Equal(t, int64(2048), params.MaxTokens)
	})

	t.Run("no budget means no thinking", func(t *testing.T) {
		params, err := anthropicParams(messages, nil, CompletionOption{
			Model:     "m",
			Reasoning: &ReasoningOption{Effort: ReasoningEffortHigh},
		})
		require.NoError(t, err)
		assert.Nil(t, params.Thinking.OfEnabled)
	})
}
