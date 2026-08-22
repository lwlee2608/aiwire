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

var (
	_ VideoGeneration = (*Service)(nil)
	_ VideoJobs       = (*Service)(nil)
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

// VideoJobs exposes video generation as a job the caller polls itself.
type VideoJobs interface {
	SubmitVideo(ctx context.Context, opt VideoOption) (VideoJob, error)
	PollVideo(ctx context.Context, jobID string) (VideoStatus, error)
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

type VideoJob struct {
	ID     string
	Status string
}

// VideoJobError reports that a video job reached a terminal state and will
// never produce a clip. A [VideoJobs] caller polling in a loop should stop on
// this error; any other error may be transient and is safe to retry.
type VideoJobError struct {
	Status  string
	Message string
}

func (e *VideoJobError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("aiwire: video generation %s: %s", e.Status, e.Message)
	}
	return fmt.Sprintf("aiwire: video generation %s", e.Status)
}

type VideoStatus struct {
	Done     bool
	Status   string
	Videos   []GeneratedVideo
	Provider string
	Usage    Usage
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
	job, err := s.SubmitVideo(ctx, opt)
	if err != nil {
		return VideoResponse{}, err
	}

	interval := opt.PollInterval
	if interval <= 0 {
		interval = videoPollInterval
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		status, err := s.PollVideo(ctx, job.ID)
		if err != nil {
			return VideoResponse{}, err
		}
		if status.Done {
			return VideoResponse{
				Videos:   status.Videos,
				Provider: status.Provider,
				Usage:    status.Usage,
			}, nil
		}

		select {
		case <-ctx.Done():
			return VideoResponse{}, ctx.Err()
		case <-ticker.C:
		}
	}
}

func (s *Service) SubmitVideo(ctx context.Context, opt VideoOption) (VideoJob, error) {
	var submit struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := s.client.Post(ctx, "videos", videoRequestBody(opt), &submit); err != nil {
		return VideoJob{}, err
	}
	if submit.ID == "" {
		return VideoJob{}, errors.New("aiwire: video generation returned no job id")
	}

	return VideoJob{ID: submit.ID, Status: submit.Status}, nil
}

func (s *Service) PollVideo(ctx context.Context, jobID string) (VideoStatus, error) {
	if jobID == "" {
		return VideoStatus{}, errors.New("aiwire: video job id is empty")
	}

	var response *http.Response
	var result struct {
		Status       string                 `json:"status"`
		UnsignedURLs []string               `json:"unsigned_urls"`
		Error        string                 `json:"error"`
		Provider     string                 `json:"provider"`
		Usage        openai.CompletionUsage `json:"usage"`
	}
	if err := s.client.Get(ctx, "videos/"+jobID, nil, &result, option.WithResponseInto(&response)); err != nil {
		return VideoStatus{}, err
	}

	switch result.Status {
	case "completed":
		videos := make([]GeneratedVideo, 0, len(result.UnsignedURLs))
		for _, u := range result.UnsignedURLs {
			if u != "" {
				videos = append(videos, GeneratedVideo{URL: u})
			}
		}
		if len(videos) == 0 {
			return VideoStatus{}, &VideoJobError{Status: result.Status, Message: "returned no video URLs"}
		}
		provider := strings.TrimSpace(result.Provider)
		if provider == "" {
			provider = extractProviderFromHeader(response)
		}
		return VideoStatus{
			Done:     true,
			Status:   result.Status,
			Videos:   videos,
			Provider: provider,
			Usage:    UsageFromOpenAI(result.Usage),
		}, nil
	case "failed", "cancelled", "expired":
		return VideoStatus{}, &VideoJobError{Status: result.Status, Message: result.Error}
	}

	return VideoStatus{Status: result.Status}, nil
}

func videoRequestBody(opt VideoOption) map[string]any {
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

	return body
}
