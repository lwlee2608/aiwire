package aiwire

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// videoPollInterval is the default delay between polls while a video job runs.
// Callers can override it via VideoOption.PollInterval.
const videoPollInterval = 5 * time.Second

// VideoGeneration generates videos from a text prompt and optional frame
// images (image-to-video). It is separate from [ImageGeneration] and
// [Completion] because video generation is asynchronous: OpenRouter's /videos
// API submits a job and the result is polled until completion.
type VideoGeneration interface {
	GenerateVideo(ctx context.Context, opt VideoOption) (VideoResponse, error)
}

// VideoFrameType selects which end of the clip a frame image anchors.
type VideoFrameType string

const (
	VideoFrameFirst VideoFrameType = "first_frame"
	VideoFrameLast  VideoFrameType = "last_frame"
)

// VideoOption configures a video-generation request.
type VideoOption struct {
	Model       string
	Prompt      string
	FrameImages []VideoFrameImage // optional first/last frames for image-to-video
	Duration    int               // clip length in seconds
	AspectRatio string            // e.g. "16:9", "9:16", "1:1"
	Resolution  string            // e.g. "1080p"
	ConfigExtra map[string]any    // extra top-level knobs (e.g. cfg_scale)

	// PollInterval overrides how often the job is polled for completion.
	// Defaults to videoPollInterval when zero.
	PollInterval time.Duration
}

// VideoFrameImage is a source frame supplied for image-to-video generation.
type VideoFrameImage struct {
	URL       string         // data URL ("data:image/png;base64,...") or a remote URL
	FrameType VideoFrameType // defaults to first_frame
}

// VideoFrameFromBytes builds a VideoFrameImage as a base64 data URL.
func VideoFrameFromBytes(mimeType string, data []byte, frameType VideoFrameType) VideoFrameImage {
	return VideoFrameImage{
		URL:       "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data),
		FrameType: frameType,
	}
}

// VideoResponse is the result of a completed video-generation request.
type VideoResponse struct {
	Videos   []GeneratedVideo
	Provider string
	Usage    Usage
}

// GeneratedVideo is one video emitted by a video-generation model.
type GeneratedVideo struct {
	URL string // remote URL to the rendered clip
}

// GenerateVideo implements [VideoGeneration]. It submits a job to OpenRouter's
// /videos endpoint, polls until the job completes, then returns the rendered
// clip URLs. It blocks until completion, failure, or ctx cancellation.
func (s *Service) GenerateVideo(ctx context.Context, opt VideoOption) (VideoResponse, error) {
	body := map[string]any{
		"model":  opt.Model,
		"prompt": opt.Prompt,
	}
	if len(opt.FrameImages) > 0 {
		frames := make([]map[string]any, 0, len(opt.FrameImages))
		for _, frame := range opt.FrameImages {
			frameType := frame.FrameType
			if frameType == "" {
				frameType = VideoFrameFirst
			}
			frames = append(frames, map[string]any{
				"type":       "image_url",
				"image_url":  map[string]string{"url": frame.URL},
				"frame_type": string(frameType),
			})
		}
		body["frame_images"] = frames
	}
	if opt.Duration > 0 {
		body["duration"] = opt.Duration
	}
	setImageParameter(body, "aspect_ratio", opt.AspectRatio)
	setImageParameter(body, "resolution", opt.Resolution)
	maps.Copy(body, opt.ConfigExtra)

	var submit struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := s.client.Post(ctx, "videos", body, &submit); err != nil {
		return VideoResponse{}, err
	}
	if submit.ID == "" {
		return VideoResponse{}, errors.New("aiwire: video generation returned no job id")
	}

	interval := opt.PollInterval
	if interval <= 0 {
		interval = videoPollInterval
	}
	return s.pollVideo(ctx, submit.ID, interval)
}

func (s *Service) pollVideo(ctx context.Context, jobID string, interval time.Duration) (VideoResponse, error) {
	path := "videos/" + jobID
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		var response *http.Response
		var result struct {
			Status       string                 `json:"status"`
			UnsignedURLs []string               `json:"unsigned_urls"`
			Error        string                 `json:"error"`
			Provider     string                 `json:"provider"`
			Usage        openai.CompletionUsage `json:"usage"`
		}
		if err := s.client.Get(ctx, path, nil, &result, option.WithResponseInto(&response)); err != nil {
			return VideoResponse{}, err
		}

		switch result.Status {
		case "completed":
			videos := make([]GeneratedVideo, 0, len(result.UnsignedURLs))
			for _, u := range result.UnsignedURLs {
				if u != "" {
					videos = append(videos, GeneratedVideo{URL: u})
				}
			}
			provider := strings.TrimSpace(result.Provider)
			if provider == "" {
				provider = extractProviderFromHeader(response)
			}
			return VideoResponse{
				Videos:   videos,
				Provider: provider,
				Usage:    UsageFromOpenAI(result.Usage),
			}, nil
		case "failed", "cancelled", "expired":
			if result.Error != "" {
				return VideoResponse{}, fmt.Errorf("aiwire: video generation %s: %s", result.Status, result.Error)
			}
			return VideoResponse{}, fmt.Errorf("aiwire: video generation %s", result.Status)
		}

		select {
		case <-ctx.Done():
			return VideoResponse{}, ctx.Err()
		case <-ticker.C:
		}
	}
}
