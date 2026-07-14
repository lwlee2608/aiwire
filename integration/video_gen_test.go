//go:build integration

package integration

import (
	"context"
	_ "embed"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/kitten.webp
var kittenWebP []byte

func TestOpenRouter_VideoGeneration(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	tests := []struct {
		model    string
		duration int
	}{
		{model: "x-ai/grok-imagine-video", duration: 1},
		// {model: "alibaba/wan-2.7", duration: 2},
		// {model: "kwaivgi/kling-v3.0-std", duration: 3},
		// {model: "kwaivgi/kling-v3.0-pro", duration: 3},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			resp, err := service.GenerateVideo(ctx, aiwire.VideoOption{
				Model:       tt.model,
				Prompt:      "The kitten gently walks toward the camera.",
				Duration:    tt.duration,
				AspectRatio: "1:1",
				FrameImages: []aiwire.VideoFrameImage{
					aiwire.VideoFrameFromBytes("image/webp", kittenWebP, aiwire.VideoFrameFirst),
				},
				ConfigExtra: map[string]any {
					"generate_audio": false,
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
		})
	}
}
