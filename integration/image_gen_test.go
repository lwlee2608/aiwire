//go:build integration

package integration

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// imageGenCase pairs a model with its output modalities. Most models emit
// both image and text (nil = aiwire's default); image-only models like
// grok-imagine must request just ["image"] or OpenRouter returns a 404.
type imageGenCase struct {
	model      string
	modalities []string
	endpoint   aiwire.ImageEndpoint
}

var imageGenModels = []imageGenCase{
	{model: "google/gemini-2.5-flash-image"},
	{model: "openai/gpt-image-1-mini", endpoint: aiwire.ImageEndpointImages},
	{model: "openai/gpt-image-2", endpoint: aiwire.ImageEndpointImages},
	{model: "openai/gpt-5.4-image-2"},
	{model: "openai/gpt-5-image"},
	{model: "google/gemini-3.1-flash-image-preview"},
	{model: "x-ai/grok-imagine-image-quality", modalities: []string{"image"}},
	{model: "bytedance-seed/seedream-4.5", modalities: []string{"image"}},
}

// modelSlug turns a model id into a filesystem-safe name for saved images.
func modelSlug(model string) string {
	return strings.NewReplacer("/", "_", ":", "_").Replace(model)
}

// saveImage writes data to /tmp/<name>.<ext> and logs the path. ext falls back
// to "bin" when the container is unrecognized.
func saveImage(t *testing.T, name, ext string, data []byte) {
	t.Helper()
	if ext == "" {
		ext = "bin"
	}
	path := filepath.Join("/tmp", name+"."+ext)
	require.NoError(t, os.WriteFile(path, data, 0o644))
	t.Logf("Saved image to %s", path)
}

// imageMagic returns a short label for a decoded image's container, or "" if unrecognized.
func imageMagic(data []byte) string {
	switch {
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		return "png"
	case bytes.HasPrefix(data, []byte("\xff\xd8\xff")):
		return "jpeg"
	case bytes.HasPrefix(data, []byte("GIF8")):
		return "gif"
	case bytes.HasPrefix(data, []byte("RIFF")) && len(data) > 11 && bytes.Equal(data[8:12], []byte("WEBP")):
		return "webp"
	default:
		return ""
	}
}

func TestOpenRouter_ImageGeneration(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	for _, tc := range imageGenModels {
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			resp, err := service.GenerateImage(context.Background(), aiwire.ImageOption{
				Model:       tc.model,
				Prompt:      "A Warhammer 40k Sister of Battle in ornate black and red power armor, wielding a flaming bolter, standing in a gothic cathedral lit by candlelight, dramatic cinematic lighting.",
				Endpoint:    tc.endpoint,
				AspectRatio: "1:1",
				Modalities:  tc.modalities,
			})
			require.NoError(t, err)
			require.NotEmpty(t, resp.Images, "expected at least one generated image")

			t.Logf("Text: %s", resp.Text)
			t.Logf("Provider: %s", resp.Provider)
			t.Logf("Images: %d", len(resp.Images))
			logUsage(t, resp.Usage)

			mime, data, err := resp.Images[0].Decode()
			require.NoError(t, err)
			assert.NotEmpty(t, data)
			kind := imageMagic(data)
			t.Logf("First image: mime=%s bytes=%d kind=%s", mime, len(data), kind)
			assert.NotEmpty(t, kind, "decoded data should be a recognizable image")
			saveImage(t, "aiwire_image_generation_"+modelSlug(tc.model), kind, data)
		})
	}
}

func TestOpenRouter_ImageEditing(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	for _, tc := range imageGenModels {
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			// ocrBase64PNG is embedded in ocr_test.go (same package): a 300x80 PNG.
			resp, err := service.GenerateImage(context.Background(), aiwire.ImageOption{
				Model:    tc.model,
				Prompt:   "Add a bright yellow border around this image.",
				Endpoint: tc.endpoint,
				Images: []aiwire.ImageInput{
					aiwire.ImageInputFromBytes("image/png", ocrBase64PNG),
				},
				Modalities: tc.modalities,
			})
			require.NoError(t, err)
			require.NotEmpty(t, resp.Images, "expected at least one edited image")

			t.Logf("Provider: %s", resp.Provider)
			logUsage(t, resp.Usage)

			mime, data, err := resp.Images[0].Decode()
			require.NoError(t, err)
			assert.NotEmpty(t, data)
			kind := imageMagic(data)
			t.Logf("Edited image: mime=%s bytes=%d kind=%s", mime, len(data), kind)
			assert.NotEmpty(t, kind, "decoded data should be a recognizable image")
			saveImage(t, "aiwire_image_editing_"+modelSlug(tc.model), kind, data)
		})
	}
}
