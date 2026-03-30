package aiwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go/v3"
)

var ErrMaxStepsExceeded = errors.New("max steps exceeded")

type ToolResult interface {
	Content() string
	Error() error
}

type SimpleToolResult struct {
	ToolContent string
	ToolError   error
}

func (t *SimpleToolResult) Content() string {
	return t.ToolContent
}

func (t *SimpleToolResult) Error() error {
	return t.ToolError
}

type ToolName string

type Tool interface {
	Definition() openai.ChatCompletionToolUnionParam
	Execute(ctx context.Context, inputs map[string]any) (ToolResult, error)
}

type AgentResult struct {
	Content     string
	Provider    string
	ToolCalls   []openai.ChatCompletionMessageToolCallUnion
	ToolResults []ToolResult
	Usage       openai.CompletionUsage
}

type Agent struct {
	aiService Completion
	maxSteps  int
}

func NewAgent(aiService Completion, maxSteps int) *Agent {
	return &Agent{
		aiService: aiService,
		maxSteps:  maxSteps,
	}
}

type PreTool struct {
	Content string `json:"content"`
	Name    string `json:"name"`
	Args    string `json:"args"`
}

type PreToolCallback func(PreTool)

type PostTool struct {
	Result ToolResult `json:"result"`
}

type PostToolCallback func(PostTool)

func (a *Agent) executeToolCall(
	ctx context.Context,
	toolCall openai.ChatCompletionMessageToolCallUnion,
	toolsMap map[string]Tool,
) (toolResult ToolResult, toolMsg openai.ChatCompletionMessageParamUnion) {
	tool, ok := toolsMap[toolCall.Function.Name]
	if !ok {
		toolResult = &SimpleToolResult{
			ToolContent: fmt.Sprintf("Tool %s not found", toolCall.Function.Name),
		}
		toolMsg = openai.ToolMessage(toolResult.Content(), toolCall.ID)
		return
	}

	var args map[string]any
	err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
	if err != nil {
		toolResult = &SimpleToolResult{
			ToolContent: fmt.Sprintf("Failed to unmarshal arguments for tool %s: %v", toolCall.Function.Name, err),
			ToolError:   err,
		}
		toolMsg = openai.ToolMessage(toolResult.Content(), toolCall.ID)
		return
	}

	toolResult, err = tool.Execute(ctx, args)
	if err != nil {
		toolResult = &SimpleToolResult{
			ToolContent: fmt.Sprintf("Error executing tool %s: %v", toolCall.Function.Name, err),
			ToolError:   err,
		}
	}
	toolMsg = openai.ToolMessage(toolResult.Content(), toolCall.ID)
	return
}

func addCompletionUsage(dst *openai.CompletionUsage, src openai.CompletionUsage) {
	dst.PromptTokens += src.PromptTokens
	dst.CompletionTokens += src.CompletionTokens
	dst.TotalTokens += src.TotalTokens

	dst.CompletionTokensDetails.AcceptedPredictionTokens += src.CompletionTokensDetails.AcceptedPredictionTokens
	dst.CompletionTokensDetails.AudioTokens += src.CompletionTokensDetails.AudioTokens
	dst.CompletionTokensDetails.ReasoningTokens += src.CompletionTokensDetails.ReasoningTokens
	dst.CompletionTokensDetails.RejectedPredictionTokens += src.CompletionTokensDetails.RejectedPredictionTokens

	dst.PromptTokensDetails.AudioTokens += src.PromptTokensDetails.AudioTokens
	dst.PromptTokensDetails.CachedTokens += src.PromptTokensDetails.CachedTokens
}

func (a *Agent) Execute(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []Tool,
	option CompletionOption,
	preCallback PreToolCallback,
	postCallback PostToolCallback,
) (*AgentResult, error) {
	return a.execute(ctx, messages, tools, option, nil, preCallback, postCallback)
}

// ExecuteStream performs the agent loop with streaming support
func (a *Agent) ExecuteStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []Tool,
	option CompletionOption,
	streamCallback StreamCallback,
	preCallback PreToolCallback,
	postCallback PostToolCallback,
) (*AgentResult, error) {
	return a.execute(ctx, messages, tools, option, streamCallback, preCallback, postCallback)
}

func (a *Agent) execute(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []Tool,
	option CompletionOption,
	streamCallback StreamCallback,
	preCallback PreToolCallback,
	postCallback PostToolCallback,
) (*AgentResult, error) {

	toolCalls := make([]openai.ChatCompletionMessageToolCallUnion, 0)
	toolResults := make([]ToolResult, 0)
	responseContent := ""
	responseProvider := ""
	totalUsage := openai.CompletionUsage{}

	toolDefinitions := make([]openai.ChatCompletionToolUnionParam, 0)
	toolsMap := make(map[string]Tool, 0)
	for _, tool := range tools {
		function := tool.Definition().GetFunction()
		if function != nil {
			toolsMap[function.Name] = tool
		}
		toolDefinitions = append(toolDefinitions, tool.Definition())
	}

	for currentStep := 0; currentStep < a.maxSteps; currentStep++ {
		var stepContent string
		var stepProvider string
		var stepToolCalls []openai.ChatCompletionMessageToolCallUnion
		var stepUsage openai.CompletionUsage

		if streamCallback != nil {
			err := a.aiService.CompletionsStream(
				ctx,
				messages,
				toolDefinitions,
				option,
				func(chunk StreamChunk) error {
					if chunk.Content != "" {
						stepContent += chunk.Content
					}
					if chunk.Provider != "" {
						stepProvider = chunk.Provider
					}
					if len(chunk.ToolCalls) > 0 {
						stepToolCalls = chunk.ToolCalls
					}
					if chunk.Done && chunk.Usage != nil {
						stepUsage = *chunk.Usage
					}
					// Don't forward Done for intermediate completions
					if chunk.Done {
						return nil
					}
					return streamCallback(chunk)
				},
			)
			if err != nil {
				return nil, err
			}
		} else {
			completion, err := a.aiService.Completions(ctx, messages, toolDefinitions, option)
			if err != nil {
				return nil, err
			}
			stepContent = completion.Message.Content
			stepProvider = completion.Provider
			stepToolCalls = completion.Message.ToolCalls
			stepUsage = completion.Usage
		}

		addCompletionUsage(&totalUsage, stepUsage)
		if stepProvider != "" {
			responseProvider = stepProvider
		}

		assistantMessage := openai.ChatCompletionMessage{
			Role:      "assistant",
			Content:   stepContent,
			ToolCalls: stepToolCalls,
		}
		messages = append(messages, assistantMessage.ToParam())

		if len(stepToolCalls) == 0 {
			if streamCallback != nil {
				streamCallback(StreamChunk{Done: true, Provider: responseProvider, Usage: &totalUsage})
			}
			return &AgentResult{
				Content:     stepContent,
				Provider:    responseProvider,
				ToolCalls:   toolCalls,
				ToolResults: toolResults,
				Usage:       totalUsage,
			}, nil
		}

		for _, toolCall := range stepToolCalls {
			if preCallback != nil {
				preCallback(PreTool{
					Content: stepContent,
					Name:    toolCall.Function.Name,
					Args:    toolCall.Function.Arguments,
				})
			}

			toolResult, toolMsg := a.executeToolCall(ctx, toolCall, toolsMap)

			if postCallback != nil {
				postCallback(PostTool{Result: toolResult})
			}

			if streamCallback != nil {
				streamCallback(StreamChunk{
					Provider: responseProvider,
					ToolResults: []StreamToolResult{
						{
							Name:    toolCall.Function.Name,
							Content: toolResult.Content(),
							Error:   toolResult.Error(),
						},
					},
				})
			}

			messages = append(messages, toolMsg)
			toolCalls = append(toolCalls, toolCall)
			toolResults = append(toolResults, toolResult)
		}

		responseContent = stepContent
	}

	if streamCallback != nil {
		streamCallback(StreamChunk{Done: true, Provider: responseProvider, Usage: &totalUsage})
	}

	return &AgentResult{
		Content:     responseContent,
		Provider:    responseProvider,
		ToolCalls:   toolCalls,
		ToolResults: toolResults,
		Usage:       totalUsage,
	}, ErrMaxStepsExceeded
}
