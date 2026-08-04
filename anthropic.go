package aiwire

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
)

const anthropicVersion = "bedrock-2023-05-31"
const anthropicDefaultMaxTokens = 4096

// AnthropicService is a minimal client for Anthropic models on AWS Bedrock,
// authenticated with a Bedrock API key (bearer token). Streaming and model
// listing are not yet supported.
type AnthropicService struct {
	apiKey     string
	region     string
	httpClient *http.Client
}

func NewAnthropicService(apiKey string, region string) *AnthropicService {
	return &AnthropicService{
		apiKey:     apiKey,
		region:     region,
		httpClient: &http.Client{},
	}
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequest struct {
	AnthropicVersion string             `json:"anthropic_version"`
	MaxTokens        int                `json:"max_tokens"`
	System           string             `json:"system,omitempty"`
	Messages         []anthropicMessage `json:"messages"`
	Temperature      *float64           `json:"temperature,omitempty"`
}

type anthropicResponse struct {
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

func (s *AnthropicService) Completions(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption) (CompletionResponse, error) {

	if len(tools) > 0 {
		return CompletionResponse{}, errors.New("anthropic: tools are not supported")
	}

	body := anthropicRequest{
		AnthropicVersion: anthropicVersion,
		MaxTokens:        anthropicDefaultMaxTokens,
	}
	if option.MaxTokens != nil {
		body.MaxTokens = *option.MaxTokens
	}
	if !option.OmitTemperature {
		body.Temperature = &option.Temperature
	}

	for _, m := range messages {
		switch {
		case m.OfSystem != nil:
			body.System = m.OfSystem.Content.OfString.Value
		case m.OfUser != nil:
			body.Messages = append(body.Messages, anthropicMessage{Role: "user", Content: m.OfUser.Content.OfString.Value})
		case m.OfAssistant != nil:
			body.Messages = append(body.Messages, anthropicMessage{Role: "assistant", Content: m.OfAssistant.Content.OfString.Value})
		default:
			return CompletionResponse{}, errors.New("anthropic: only system, user, and assistant text messages are supported")
		}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return CompletionResponse{}, err
	}

	endpoint := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke",
		s.region, url.PathEscape(option.Model))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return CompletionResponse{}, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return CompletionResponse{}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return CompletionResponse{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return CompletionResponse{}, fmt.Errorf("anthropic: %s: %s", resp.Status, respBody)
	}

	var out anthropicResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return CompletionResponse{}, err
	}

	var content string
	for _, c := range out.Content {
		if c.Type == "text" {
			content += c.Text
		}
	}

	return CompletionResponse{
		Message: openai.ChatCompletionMessage{
			Role:    "assistant",
			Content: content,
		},
		Provider: "anthropic",
		Usage: Usage{
			PromptTokens:     out.Usage.InputTokens,
			CompletionTokens: out.Usage.OutputTokens,
			TotalTokens:      out.Usage.InputTokens + out.Usage.OutputTokens,
			PromptTokensDetails: PromptTokensDetails{
				CachedTokens:        out.Usage.CacheReadInputTokens,
				CacheCreationTokens: out.Usage.CacheCreationInputTokens,
			},
		},
	}, nil
}

func (s *AnthropicService) CompletionsStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
	callback StreamCallback) error {
	return errors.New("anthropic: streaming is not supported")
}

func (s *AnthropicService) Models(ctx context.Context) (*pagination.Page[openai.Model], error) {
	return nil, errors.New("anthropic: model listing is not supported")
}

var _ Completion = (*AnthropicService)(nil)
