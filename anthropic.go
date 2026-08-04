package aiwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
)

const anthropicDefaultMaxTokens = 4096

const anthropicProvider = "anthropic"

// OpenRouter's wire shape, so details round-trip through
// [AssistantMessageWithReasoning] like every other provider's.
const (
	reasoningDetailTypeText      = "reasoning.text"
	reasoningDetailTypeEncrypted = "reasoning.encrypted"
)

// AnthropicService reaches Claude through the native Messages API, exchanging
// the OpenAI-shaped types used elsewhere in this package. No model listing.
type AnthropicService struct {
	client anthropic.Client
}

// NewAnthropicService targets AWS Bedrock in region, authenticated with a
// Bedrock API key (bearer token).
func NewAnthropicService(apiKey string, region string) *AnthropicService {
	cfg := aws.Config{
		Region:                  region,
		BearerAuthTokenProvider: bedrock.NewStaticBearerTokenProvider(apiKey),
	}
	return &AnthropicService{
		// WithoutEnvironmentDefaults stops a stray ANTHROPIC_API_KEY from
		// riding along as an X-Api-Key header.
		client: anthropic.NewClient(
			anthropicoption.WithoutEnvironmentDefaults(),
			bedrock.WithConfig(cfg),
		),
	}
}

// NewAnthropicAPIService targets the first-party Anthropic API.
func NewAnthropicAPIService(apiKey string) *AnthropicService {
	return &AnthropicService{
		client: anthropic.NewClient(anthropicoption.WithAPIKey(apiKey)),
	}
}

func (s *AnthropicService) Completions(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption) (CompletionResponse, error) {

	params, err := anthropicParams(messages, tools, option)
	if err != nil {
		return CompletionResponse{}, err
	}

	message, err := s.client.Messages.New(ctx, params)
	if err != nil {
		return CompletionResponse{}, err
	}

	out := openai.ChatCompletionMessage{Role: "assistant"}
	var content, reasoning strings.Builder
	var details []ReasoningDetail

	for _, block := range message.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "thinking":
			reasoning.WriteString(block.Thinking)
			details = append(details, anthropicReasoningText(len(details), block.Thinking, block.Signature))
		case "redacted_thinking":
			details = append(details, anthropicReasoningEncrypted(len(details), block.Data))
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, openai.ChatCompletionMessageToolCallUnion{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageFunctionToolCallFunction{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		}
	}
	out.Content = content.String()

	return CompletionResponse{
		Message:          out,
		Reasoning:        reasoning.String(),
		ReasoningDetails: details,
		Provider:         anthropicProvider,
		Usage:            anthropicUsage(message.Usage),
	}, nil
}

func (s *AnthropicService) CompletionsStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
	callback StreamCallback) error {

	params, err := anthropicParams(messages, tools, option)
	if err != nil {
		return err
	}

	stream := s.client.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	acc := anthropicStreamAccum{blocks: map[int64]*anthropicStreamBlock{}}
	var usage Usage
	var inputTokens int64

	for stream.Next() {
		event := stream.Current()
		chunk := StreamChunk{Provider: anthropicProvider}

		switch event.Type {
		case "message_start":
			chunk.Role = "assistant"
			usage = anthropicUsage(event.Message.Usage)
			inputTokens = event.Message.Usage.InputTokens
		case "content_block_start":
			acc.start(event.Index, event.ContentBlock)
			if event.ContentBlock.Type == "tool_use" {
				chunk.ToolCalls = acc.toolCalls()
			}
		case "content_block_delta":
			block := acc.blocks[event.Index]
			switch event.Delta.Type {
			case "text_delta":
				chunk.Content = event.Delta.Text
			case "thinking_delta":
				chunk.Reasoning = event.Delta.Thinking
				if block != nil {
					block.thinking.WriteString(event.Delta.Thinking)
				}
			case "signature_delta":
				if block != nil {
					block.signature += event.Delta.Signature
				}
			case "input_json_delta":
				if block != nil && block.toolCall != nil {
					block.toolCall.Function.Arguments += event.Delta.PartialJSON
					chunk.ToolCalls = acc.toolCalls()
				}
			}
		case "message_delta":
			chunk.FinishReason = anthropicFinishReason(string(event.Delta.StopReason))
			usage.CompletionTokens = event.Usage.OutputTokens
			if event.Usage.InputTokens > 0 {
				inputTokens = event.Usage.InputTokens
			}
			if event.Usage.CacheReadInputTokens > 0 {
				usage.PromptTokensDetails.CachedTokens = event.Usage.CacheReadInputTokens
			}
			if event.Usage.CacheCreationInputTokens > 0 {
				usage.PromptTokensDetails.CacheCreationTokens = event.Usage.CacheCreationInputTokens
			}
			usage.CompletionTokensDetails.ReasoningTokens = event.Usage.OutputTokensDetails.ThinkingTokens
			usage.PromptTokens = anthropicPromptTokens(
				inputTokens,
				usage.PromptTokensDetails.CachedTokens,
				usage.PromptTokensDetails.CacheCreationTokens,
			)
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		}

		if chunk.Content == "" && chunk.Reasoning == "" && chunk.Role == "" &&
			chunk.FinishReason == "" && len(chunk.ToolCalls) == 0 {
			continue
		}
		if err := callback(chunk); err != nil {
			return err
		}
	}

	if err := stream.Err(); err != nil {
		return err
	}

	return callback(StreamChunk{
		Done:             true,
		Provider:         anthropicProvider,
		Usage:            &usage,
		ReasoningDetails: acc.reasoningDetails(),
	})
}

func (s *AnthropicService) Models(ctx context.Context) (*pagination.Page[openai.Model], error) {
	return nil, errors.New("anthropic: model listing is not supported")
}

func anthropicParams(
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
) (anthropic.MessageNewParams, error) {
	system, converted, err := anthropicMessages(messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	toolParams, err := anthropicTools(tools)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	outputConfig, err := anthropicOutputConfig(option.ResponseFormat)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	params := anthropic.MessageNewParams{
		Model:        anthropic.Model(option.Model),
		MaxTokens:    anthropicDefaultMaxTokens,
		System:       system,
		Messages:     converted,
		Tools:        toolParams,
		OutputConfig: outputConfig,
	}
	if option.MaxTokens != nil {
		params.MaxTokens = int64(*option.MaxTokens)
	}
	if option.Reasoning != nil {
		anthropicThinking(&params, *option.Reasoning, option.MaxTokens == nil)
	}
	// Thinking, adaptive included, pins temperature to 1; sending both is a
	// request error.
	if !option.OmitTemperature && params.Thinking.OfEnabled == nil && params.Thinking.OfAdaptive == nil {
		params.Temperature = anthropic.Float(option.Temperature)
	}

	return params, nil
}

// Anthropic splits thinking control across two mutually exclusive mechanisms,
// by model generation: 4.5-era models take thinking.enabled with an explicit
// budget, Claude 5 models take thinking.adaptive plus output_config.effort and
// reject a budget. Each 400s on the other's models, so MaxTokens and Effort
// pick the mechanism and MaxTokens wins when both are set.
func anthropicThinking(params *anthropic.MessageNewParams, reasoning ReasoningOption, defaultMaxTokens bool) {
	switch {
	case reasoning.MaxTokens != nil:
		budget := int64(*reasoning.MaxTokens)
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: budget},
		}
		// max_tokens must exceed the thinking budget; the default may not.
		// An explicit MaxTokens is left alone.
		if defaultMaxTokens && params.MaxTokens <= budget {
			params.MaxTokens = budget + anthropicDefaultMaxTokens
		}
	case reasoning.Effort == ReasoningEffortNone:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{},
		}
	case reasoning.Effort != "":
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}
		params.OutputConfig.Effort = anthropicEffort(reasoning.Effort)
	}
}

// Anthropic has no "minimal" tier, so it floors to low.
func anthropicEffort(effort ReasoningEffort) anthropic.OutputConfigEffort {
	switch effort {
	case ReasoningEffortXHigh:
		return anthropic.OutputConfigEffortXhigh
	case ReasoningEffortHigh:
		return anthropic.OutputConfigEffortHigh
	case ReasoningEffortMedium:
		return anthropic.OutputConfigEffortMedium
	}
	return anthropic.OutputConfigEffortLow
}

func anthropicOutputConfig(format openai.ChatCompletionNewParamsResponseFormatUnion) (anthropic.OutputConfigParam, error) {
	switch {
	case format.OfJSONSchema != nil:
		schema, err := anthropicSchemaMap(format.OfJSONSchema.JSONSchema.Schema)
		if err != nil {
			return anthropic.OutputConfigParam{}, err
		}
		return anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		}, nil
	case format.OfJSONObject != nil:
		return anthropic.OutputConfigParam{}, errors.New("anthropic: json_object response format is not supported, use json_schema")
	}
	return anthropic.OutputConfigParam{}, nil
}

func anthropicSchemaMap(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, errors.New("anthropic: json_schema response format requires a schema")
	}
	if m, ok := schema.(map[string]any); ok {
		return m, nil
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("anthropic: invalid response format schema: %w", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("anthropic: invalid response format schema: %w", err)
	}
	return out, nil
}

func anthropicMessages(messages []openai.ChatCompletionMessageParamUnion) ([]anthropic.TextBlockParam, []anthropic.MessageParam, error) {
	var system []anthropic.TextBlockParam
	var out []anthropic.MessageParam

	add := func(role anthropic.MessageParamRole, blocks ...anthropic.ContentBlockParamUnion) {
		if len(blocks) == 0 {
			return
		}
		// One turn per role: consecutive same-role messages, parallel tool
		// results especially, must merge.
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			return
		}
		out = append(out, anthropic.MessageParam{Role: role, Content: blocks})
	}

	for _, m := range messages {
		switch {
		case m.OfSystem != nil:
			text, err := anthropicTextContent(m.OfSystem.Content)
			if err != nil {
				return nil, nil, err
			}
			system = append(system, anthropic.TextBlockParam{Text: text})
		case m.OfDeveloper != nil:
			text, err := anthropicTextContent(m.OfDeveloper.Content)
			if err != nil {
				return nil, nil, err
			}
			system = append(system, anthropic.TextBlockParam{Text: text})
		case m.OfUser != nil:
			text, err := anthropicTextContent(m.OfUser.Content)
			if err != nil {
				return nil, nil, err
			}
			add(anthropic.MessageParamRoleUser, anthropic.NewTextBlock(text))
		case m.OfAssistant != nil:
			blocks, err := anthropicAssistantBlocks(*m.OfAssistant)
			if err != nil {
				return nil, nil, err
			}
			add(anthropic.MessageParamRoleAssistant, blocks...)
		case m.OfTool != nil:
			text, err := anthropicTextContent(m.OfTool.Content)
			if err != nil {
				return nil, nil, err
			}
			add(anthropic.MessageParamRoleUser, anthropic.NewToolResultBlock(m.OfTool.ToolCallID, text, false))
		default:
			return nil, nil, errors.New("anthropic: unsupported message type")
		}
	}

	return system, out, nil
}

func anthropicAssistantBlocks(m openai.ChatCompletionAssistantMessageParam) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := anthropicThinkingBlocks(m)

	text, err := anthropicTextContent(m.Content)
	if err != nil {
		return nil, err
	}
	if text != "" {
		blocks = append(blocks, anthropic.NewTextBlock(text))
	}

	for _, call := range m.ToolCalls {
		fn := call.OfFunction
		if fn == nil {
			return nil, errors.New("anthropic: only function tool calls are supported")
		}
		input := any(map[string]any{})
		if args := fn.Function.Arguments; args != "" {
			if err := json.Unmarshal([]byte(args), &input); err != nil {
				return nil, fmt.Errorf("anthropic: tool call %s has invalid arguments: %w", fn.ID, err)
			}
		}
		blocks = append(blocks, anthropic.NewToolUseBlock(fn.ID, input, fn.Function.Name))
	}

	return blocks, nil
}

// anthropicThinkingBlocks recovers thinking blocks stashed by
// [AssistantMessageWithReasoning]. They must lead the turn and keep their
// signature, or Anthropic rejects the follow-up.
func anthropicThinkingBlocks(m openai.ChatCompletionAssistantMessageParam) []anthropic.ContentBlockParamUnion {
	raw, err := m.MarshalJSON()
	if err != nil {
		return nil
	}

	var probe struct {
		ReasoningDetails []struct {
			Type      string `json:"type"`
			Text      string `json:"text"`
			Data      string `json:"data"`
			Signature string `json:"signature"`
		} `json:"reasoning_details"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil
	}

	var blocks []anthropic.ContentBlockParamUnion
	for _, detail := range probe.ReasoningDetails {
		switch detail.Type {
		case reasoningDetailTypeText:
			if detail.Signature != "" {
				blocks = append(blocks, anthropic.NewThinkingBlock(detail.Signature, detail.Text))
			}
		case reasoningDetailTypeEncrypted:
			blocks = append(blocks, anthropic.NewRedactedThinkingBlock(detail.Data))
		}
	}
	return blocks
}

// anthropicTextContent flattens an OpenAI message content union. Every variant
// marshals to a JSON string or an array of content parts, so decoding the
// marshaled form covers all of them.
func anthropicTextContent(content json.Marshaler) (string, error) {
	raw, err := content.MarshalJSON()
	if err != nil {
		return "", err
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("anthropic: unsupported message content: %s", raw)
	}

	var b strings.Builder
	for _, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("anthropic: unsupported content part %q", part.Type)
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

func anthropicTools(tools []openai.ChatCompletionToolUnionParam) ([]anthropic.ToolUnionParam, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		fn := t.GetFunction()
		if fn == nil {
			return nil, errors.New("anthropic: only function tools are supported")
		}

		tool := anthropic.ToolParam{Name: fn.Name}
		if fn.Description.Valid() {
			tool.Description = anthropic.String(fn.Description.Value)
		}
		if fn.Strict.Valid() {
			tool.Strict = anthropic.Bool(fn.Strict.Value)
		}
		for key, value := range fn.Parameters {
			switch key {
			case "type":
			case "properties":
				tool.InputSchema.Properties = value
			case "required":
				tool.InputSchema.Required = anthropicStringSlice(value)
			default:
				if tool.InputSchema.ExtraFields == nil {
					tool.InputSchema.ExtraFields = map[string]any{}
				}
				tool.InputSchema.ExtraFields[key] = value
			}
		}

		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out, nil
}

func anthropicStringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func anthropicReasoningText(index int, text string, signature string) ReasoningDetail {
	raw, _ := json.Marshal(map[string]any{
		"type":      reasoningDetailTypeText,
		"index":     index,
		"text":      text,
		"signature": signature,
	})
	return ReasoningDetail{Type: reasoningDetailTypeText, Index: index, Raw: raw}
}

func anthropicReasoningEncrypted(index int, data string) ReasoningDetail {
	raw, _ := json.Marshal(map[string]any{
		"type":  reasoningDetailTypeEncrypted,
		"index": index,
		"data":  data,
	})
	return ReasoningDetail{Type: reasoningDetailTypeEncrypted, Index: index, Raw: raw}
}

func anthropicUsage(u anthropic.Usage) Usage {
	prompt := anthropicPromptTokens(u.InputTokens, u.CacheReadInputTokens, u.CacheCreationInputTokens)
	return Usage{
		PromptTokens:     prompt,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      prompt + u.OutputTokens,
		PromptTokensDetails: PromptTokensDetails{
			CachedTokens:        u.CacheReadInputTokens,
			CacheCreationTokens: u.CacheCreationInputTokens,
		},
		CompletionTokensDetails: CompletionTokensDetails{
			ReasoningTokens: u.OutputTokensDetails.ThinkingTokens,
		},
	}
}

// Anthropic reports input_tokens exclusive of cache hits and writes, while the
// normalized PromptTokens is inclusive like OpenAI's prompt_tokens.
func anthropicPromptTokens(input, cacheRead, cacheCreation int64) int64 {
	return input + cacheRead + cacheCreation
}

func anthropicFinishReason(stopReason string) string {
	switch stopReason {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	}
	return stopReason
}

// anthropicStreamBlock accumulates one content block across its delta events.
type anthropicStreamBlock struct {
	kind      string
	thinking  strings.Builder
	signature string
	data      string
	toolCall  *openai.ChatCompletionMessageToolCallUnion
}

type anthropicStreamAccum struct {
	blocks map[int64]*anthropicStreamBlock
	order  []int64
}

func (a *anthropicStreamAccum) start(index int64, block anthropic.ContentBlockStartEventContentBlockUnion) {
	entry := &anthropicStreamBlock{kind: block.Type, data: block.Data}
	if block.Type == "tool_use" {
		entry.toolCall = &openai.ChatCompletionMessageToolCallUnion{
			ID:   block.ID,
			Type: "function",
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name: block.Name,
			},
		}
	}
	a.blocks[index] = entry
	a.order = append(a.order, index)
}

func (a *anthropicStreamAccum) toolCalls() []openai.ChatCompletionMessageToolCallUnion {
	var out []openai.ChatCompletionMessageToolCallUnion
	for _, index := range a.order {
		call := a.blocks[index].toolCall
		if call == nil {
			continue
		}
		snapshot := *call
		// A no-argument tool streams no input_json_delta, which would
		// otherwise surface as unparseable empty arguments.
		if snapshot.Function.Arguments == "" {
			snapshot.Function.Arguments = "{}"
		}
		out = append(out, snapshot)
	}
	return out
}

func (a *anthropicStreamAccum) reasoningDetails() []ReasoningDetail {
	var out []ReasoningDetail
	for _, index := range a.order {
		block := a.blocks[index]
		switch block.kind {
		case "thinking":
			out = append(out, anthropicReasoningText(len(out), block.thinking.String(), block.signature))
		case "redacted_thinking":
			out = append(out, anthropicReasoningEncrypted(len(out), block.data))
		}
	}
	return out
}

var _ Completion = (*AnthropicService)(nil)
