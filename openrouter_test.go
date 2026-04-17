//go:build openrouter

package aiwire

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)


func TestOpenAI_Completion_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.7,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &ReasoningOption{
			Effort: ReasoningEffortLow,
		},
	})
}

func TestOpenAI_ResponseFormat_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	type person struct {
		Name string `json:"name" jsonschema:"required"`
		Age  int    `json:"age" jsonschema:"required"`
	}

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Return a person named Alice who is 30 years old."),
	}

	response, err := service.Completions(context.Background(), messages, nil, CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.0,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "person",
					Strict: openai.Bool(true),
					Schema: GenerateSchema[person](),
				},
			},
		},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)

	var got person
	err = json.Unmarshal([]byte(response.Message.Content), &got)
	assert.NoError(t, err)
	assert.Equal(t, "Alice", got.Name)
	assert.Equal(t, 30, got.Age)

	t.Logf("Response: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func TestOpenAI_Streaming_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.7,
		Provider: &ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &ReasoningOption{
			Effort: ReasoningEffortLow,
		},
	})
}
