//go:build integration

package integration

import (
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_Completion(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: 0.7,
	})
}

func TestOpenAI_Completion_TemperatureUnsupported(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	t.Run("rejects temperature", func(t *testing.T) {
		_, err := service.Completions(t.Context(), messages, nil, aiwire.CompletionOption{
			Model:       "gpt-5.6-sol",
			Temperature: 0.7,
		})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "temperature")
		t.Logf("Error: %v", err)
	})

	t.Run("succeeds when omitted", func(t *testing.T) {
		runCompletionTest(t, service, messages, aiwire.CompletionOption{
			Model:           "gpt-5.6-sol",
			OmitTemperature: true,
		})
	})
}

func TestOpenAI_Respond_TemperatureUnsupported(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	input := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage("Hello, can you tell me a joke?", responses.EasyInputMessageRoleUser),
	}

	t.Run("rejects temperature", func(t *testing.T) {
		_, err := service.Respond(t.Context(), input, nil, aiwire.ResponsesOption{
			Model:       "gpt-5.6-sol",
			Temperature: 0.7,
		})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "temperature")
		t.Logf("Error: %v", err)
	})

	t.Run("succeeds when omitted", func(t *testing.T) {
		resp, err := service.Respond(t.Context(), input, nil, aiwire.ResponsesOption{
			Model:           "gpt-5.6-sol",
			OmitTemperature: true,
		})
		assert.NoError(t, err)
		assert.NotEmpty(t, resp.ID)
		assert.NotEmpty(t, resp.OutputText())

		t.Logf("Response: %s", resp.OutputText())
		logUsage(t, resp.Usage)
	})
}

func TestOpenAI_Embedding(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")

	embedding, err := service.Embedding(t.Context(), "Hello, world!", "text-embedding-3-small")
	assert.NoError(t, err)
	assert.NotEmpty(t, embedding)
	assert.Equal(t, 1536, len(embedding))
}

func TestOpenAI_Respond(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	input := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage("Hello, can you tell me a joke?", responses.EasyInputMessageRoleUser),
	}

	resp, err := service.Respond(t.Context(), input, nil, aiwire.ResponsesOption{
		Model:       "gpt-4.1-nano",
		Temperature: 0.7,
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, resp.ID)

	text := resp.OutputText()
	assert.NotEmpty(t, text)

	t.Logf("Response: %s", text)
	t.Logf("Status: %s", resp.Status)
	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)
}

func TestOpenAI_Streaming(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: 0.7,
	})
}
