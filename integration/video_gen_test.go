//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/require"
)

func TestOpenRouter_VideoGeneration(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	// Cheapest config: std tier, shortest clip. ocrBase64PNG (embedded in
	// ocr_test.go) is reused as the first frame for image-to-video.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	resp, err := service.GenerateVideo(ctx, aiwire.VideoOption{
		Model:       "kwaivgi/kling-v3.0-std",
		Prompt:      "The text gently zooms in with a soft glow.",
		Duration:    5,
		AspectRatio: "1:1",
		FrameImages: []aiwire.VideoFrameImage{
			aiwire.VideoFrameFromBytes("image/png", ocrBase64PNG, aiwire.VideoFrameFirst),
		},
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Videos, "expected at least one generated video")

	t.Logf("Provider: %s", resp.Provider)
	t.Logf("Videos: %d", len(resp.Videos))
	for i, v := range resp.Videos {
		t.Logf("Video %d: %s", i, v.URL)
	}
	logUsage(t, resp.Usage)
}
