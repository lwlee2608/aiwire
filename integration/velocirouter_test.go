//go:build integration

package integration

import (
	"bytes"
	"context"
	"errors"
	"image"
	_ "image/png"
	"net/http"
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// VelociRouter serves OpenRouter's /images shape for gpt-image-2. Its backend
// picks size, quality, count, and format itself, so it accepts only aspect_ratio,
// input references, and the background/n extras.
const velocirouterImageModel = "openai/gpt-image-2"

func velocirouterService(t *testing.T) *aiwire.Service {
	t.Helper()
	baseURL := os.Getenv("VELOCIROUTER_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.velocirouter.site/v1"
	}
	return aiwire.NewOpenAIService(keyOrSkip(t, "VELOCIROUTER_API_KEY"), baseURL)
}

func TestVelociRouter_Completion(t *testing.T) {
	t.Parallel()
	runCompletionTest(t, velocirouterService(t), []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}, aiwire.CompletionOption{
		Model:           "gpt-6-luna",
		OmitTemperature: true,
	})
}

func TestVelociRouter_ImageGeneration(t *testing.T) {
	t.Parallel()
	resp, err := velocirouterService(t).GenerateImage(context.Background(), aiwire.ImageOption{
		Model:       velocirouterImageModel,
		Prompt:      "A red circle on a white background.",
		Endpoint:    aiwire.ImageEndpointImages,
		AspectRatio: "16:9",
	})
	require.NoError(t, err)
	require.Len(t, resp.Images, 1)
	logUsage(t, resp.Usage)

	mime, data, err := resp.Images[0].Decode()
	require.NoError(t, err)
	assert.Equal(t, "image/png", mime)
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	require.NoError(t, err)
	t.Logf("Generated image: %dx%d bytes=%d", cfg.Width, cfg.Height, len(data))
	assert.Greater(t, cfg.Width, cfg.Height, "16:9 should yield a landscape image")
	saveImage(t, "aiwire_velocirouter_image_generation", imageMagic(data), data)
}

func TestVelociRouter_ImageEditing(t *testing.T) {
	t.Parallel()
	// ocrBase64PNG is embedded in ocr_test.go (same package): a 300x80 PNG.
	resp, err := velocirouterService(t).GenerateImage(context.Background(), aiwire.ImageOption{
		Model:    velocirouterImageModel,
		Prompt:   "Add a bright yellow border around this image.",
		Endpoint: aiwire.ImageEndpointImages,
		Images:   []aiwire.ImageInput{aiwire.ImageInputFromBytes("image/png", ocrBase64PNG)},
	})
	require.NoError(t, err)
	require.Len(t, resp.Images, 1)
	logUsage(t, resp.Usage)

	mime, data, err := resp.Images[0].Decode()
	require.NoError(t, err)
	assert.Equal(t, "image/png", mime)
	t.Logf("Edited image: bytes=%d", len(data))
	saveImage(t, "aiwire_velocirouter_image_editing", imageMagic(data), data)
}

func TestVelociRouter_ImageRejectsUnsupportedParams(t *testing.T) {
	t.Parallel()
	service := velocirouterService(t)
	for _, tc := range []struct {
		name string
		opt  aiwire.ImageOption
	}{
		{"quality", aiwire.ImageOption{Quality: "high"}},
		{"resolution", aiwire.ImageOption{Resolution: "2K"}},
		{"output_format", aiwire.ImageOption{OutputFormat: "webp"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opt := tc.opt
			opt.Model, opt.Prompt, opt.Endpoint = velocirouterImageModel, "x", aiwire.ImageEndpointImages
			_, err := service.GenerateImage(context.Background(), opt)
			var apiErr *openai.Error
			require.True(t, errors.As(err, &apiErr), "err=%v", err)
			assert.Equal(t, http.StatusBadRequest, apiErr.StatusCode)
			assert.Equal(t, tc.name, apiErr.Param)
		})
	}
}
