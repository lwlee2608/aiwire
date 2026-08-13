//go:build integration

package integration

import (
	"context"
	_ "embed"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/kitten.webp
var kittenWebP []byte

// saveVideo downloads url to /tmp/<name>.mp4 and logs the path. OpenRouter's
// video URLs are unsigned, so the API key must be sent to fetch the content.
func saveVideo(t *testing.T, name, url, apiKey string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	data, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.NotEmpty(t, data)

	path := filepath.Join("/tmp", name+".mp4")
	require.NoError(t, os.WriteFile(path, data, 0o644))
	t.Logf("Saved video to %s (%d bytes)", path, len(data))
}

func TestOpenRouter_VideoGeneration(t *testing.T) {
	apiKey := keyOrSkip(t, "OPENROUTER_API_KEY")
	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	//
	// Measured 2026-08-13 at these settings (500x500 kitten first frame):
	//   model                        size     cost      time
	//   x-ai/grok-imagine-video      482 KB   $0.052     39s
	//   bytedance/seedance-2.0-mini  1.5 MB   $0.054    129s
	//   bytedance/seedance-2.0       1.0 MB   $0.272    129s
	//   bytedance/seedance-2.5       2.6 MB   $0.415    129s
	//   total                                 $0.793   ~7min
	tests := []struct {
		model       string
		duration    int
		resolution  string
		aspectRatio string
	}{
		{model: "x-ai/grok-imagine-video", duration: 1, resolution: "480p", aspectRatio: "1:1"},
		{model: "bytedance/seedance-2.0-mini", duration: 4, resolution: "480p", aspectRatio: "1:1"},
		{model: "bytedance/seedance-2.0", duration: 4, resolution: "480p", aspectRatio: "1:1"},
		{model: "bytedance/seedance-2.5", duration: 4, resolution: "480p"},
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
				AspectRatio: tt.aspectRatio,
				Resolution:  tt.resolution,
				FrameImages: []aiwire.VideoFrameImage{
					aiwire.VideoFrameFromBytes("image/webp", kittenWebP, aiwire.VideoFrameFirst),
				},
				ConfigExtra: map[string]any{
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
			saveVideo(t, "aiwire_video_generation_"+modelSlug(tt.model), resp.Videos[0].URL, apiKey)
			logUsage(t, resp.Usage)
		})
	}
}
