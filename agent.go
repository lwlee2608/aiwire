package aiwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"

	"github.com/openai/openai-go/v3"
)

// ErrMaxStepsExceeded is returned by Agent.Execute / Agent.ExecuteStream when
// the loop hits maxSteps without the model producing a final tool-free answer.
var ErrMaxStepsExceeded = errors.New("max steps exceeded")

// ErrToolPanic is wrapped by the ToolResult error of a tool call whose Execute
// panicked. The wrapping error holds the panic value and stack; the model only
// sees a short message.
var ErrToolPanic = errors.New("tool panicked")

// ToolResult is the value returned by a Tool.Execute call. Content is the
// string fed back to the model; Error is informational and does not abort the loop.
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

// Tool is invoked by the agent when the model emits a matching tool call.
// Definition returns the OpenAI tool schema; Execute runs the tool with the
// model-supplied arguments.
type Tool interface {
	Definition() openai.ChatCompletionToolUnionParam
	Execute(ctx context.Context, inputs map[string]any) (ToolResult, error)
}

type AgentResult struct {
	Content     string
	Provider    string
	ToolCalls   []openai.ChatCompletionMessageToolCallUnion
	ToolResults []ToolResult
	Usage       Usage
}

// Agent orchestrates a multi-step completion-and-tool-call loop against a Completion service.
type Agent struct {
	aiService Completion
	maxSteps  int
}

// NewAgent returns an Agent that drives aiService for at most maxSteps iterations per run.
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
	Name   string     `json:"name"`
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

	toolResult, panicked, err := runTool(ctx, tool, args)
	if panicked {
		toolResult = &SimpleToolResult{
			ToolContent: fmt.Sprintf("Tool %s crashed", toolCall.Function.Name),
			ToolError:   err,
		}
	} else if err != nil {
		toolResult = &SimpleToolResult{
			ToolContent: fmt.Sprintf("Error executing tool %s: %v", toolCall.Function.Name, err),
			ToolError:   err,
		}
	}
	toolMsg = openai.ToolMessage(toolResult.Content(), toolCall.ID)
	return
}

func runTool(ctx context.Context, tool Tool, args map[string]any) (result ToolResult, panicked bool, err error) {
	defer func() {
		if r := recover(); r != nil {
			result, panicked = nil, true
			err = fmt.Errorf("%w: %v\n%s", ErrToolPanic, r, debug.Stack())
		}
	}()
	result, err = tool.Execute(ctx, args)
	return
}

// Execute runs the agent loop without streaming and returns the final result.
// It returns ErrMaxStepsExceeded if the loop reaches maxSteps without a tool-free
// assistant message; the partial result is still returned in that case.
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

// ExecuteStream runs the agent loop and forwards each StreamChunk to streamCallback.
// The Done flag is only set on the final terminal chunk; intermediate completions
// suppress it so callers can treat the stream as a single conversation.
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
	totalUsage := Usage{}

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
		var stepUsage Usage
		var stepReasoningDetails []ReasoningDetail

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
					if chunk.Done && len(chunk.ReasoningDetails) > 0 {
						stepReasoningDetails = chunk.ReasoningDetails
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
			stepReasoningDetails = completion.ReasoningDetails
		}

		totalUsage.Add(stepUsage)
		if stepProvider != "" {
			responseProvider = stepProvider
		}

		messages = append(messages, AssistantMessageWithReasoning(stepContent, stepToolCalls, stepReasoningDetails))

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
				postCallback(PostTool{Name: toolCall.Function.Name, Result: toolResult})
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
