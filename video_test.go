package aiwire

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestGenerateVideoReturnsErrorWhenCompletedWithoutVideos(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			_, _ = w.Write([]byte(`{"id":"job-1","status":"pending"}`))
		case http.MethodGet:
			_, _ = w.Write([]byte(`{"id":"job-1","status":"completed","unsigned_urls":[]}`))
		}
	}))
	defer server.Close()

	_, err := NewOpenAIService("test-key", server.URL).GenerateVideo(context.Background(), VideoOption{
		Model:        "video-model",
		Prompt:       "generate a video",
		PollInterval: time.Millisecond,
	})
	if err == nil {
		t.Fatal("GenerateVideo: expected error, got nil")
	}
	if !strings.Contains(err.Error(), "no video URLs") {
		t.Errorf("error = %q, want missing video URL error", err)
	}
}

func TestGenerateVideoReturnsTerminalJobErrors(t *testing.T) {
	for _, status := range []string{"failed", "cancelled", "expired"} {
		t.Run(status, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.Method {
				case http.MethodPost:
					_, _ = w.Write([]byte(`{"id":"job-1","status":"pending"}`))
				case http.MethodGet:
					_, _ = fmt.Fprintf(w, `{"id":"job-1","status":%q,"error":"provider stopped"}`, status)
				}
			}))
			defer server.Close()

			_, err := NewOpenAIService("test-key", server.URL).GenerateVideo(context.Background(), VideoOption{
				Model:        "video-model",
				Prompt:       "generate a video",
				PollInterval: time.Millisecond,
			})
			if err == nil {
				t.Fatal("GenerateVideo: expected error, got nil")
			}
			if !strings.Contains(err.Error(), status) || !strings.Contains(err.Error(), "provider stopped") {
				t.Errorf("error = %q, want status and provider error", err)
			}
		})
	}
}

func TestSubmitVideoReturnsJobID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"job-1","status":"queued"}`))
	}))
	defer server.Close()

	job, err := NewOpenAIService("test-key", server.URL).SubmitVideo(context.Background(), VideoOption{
		Model:  "video-model",
		Prompt: "generate a video",
	})
	if err != nil {
		t.Fatalf("SubmitVideo: %v", err)
	}
	if job.ID != "job-1" {
		t.Errorf("job ID = %q, want job-1", job.ID)
	}
	if job.Status != "queued" {
		t.Errorf("job status = %q, want queued", job.Status)
	}
}

func TestSubmitVideoRequiresJobID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"queued"}`))
	}))
	defer server.Close()

	_, err := NewOpenAIService("test-key", server.URL).SubmitVideo(context.Background(), VideoOption{Model: "video-model"})
	if err == nil || !strings.Contains(err.Error(), "no job id") {
		t.Fatalf("error = %v, want missing job id error", err)
	}
}

func TestPollVideoReportsPendingWithoutWaiting(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"job-1","status":"processing"}`))
	}))
	defer server.Close()

	status, err := NewOpenAIService("test-key", server.URL).PollVideo(context.Background(), "job-1")
	if err != nil {
		t.Fatalf("PollVideo: %v", err)
	}
	if status.Done {
		t.Error("Done = true, want false while processing")
	}
	if status.Status != "processing" {
		t.Errorf("Status = %q, want processing", status.Status)
	}
	if calls != 1 {
		t.Errorf("requests = %d, want exactly 1", calls)
	}
}

func TestPollVideoReturnsVideosWhenCompleted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/videos/job-1"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"completed","unsigned_urls":["https://example.invalid/clip.mp4"],"provider":"acme"}`))
	}))
	defer server.Close()

	status, err := NewOpenAIService("test-key", server.URL).PollVideo(context.Background(), "job-1")
	if err != nil {
		t.Fatalf("PollVideo: %v", err)
	}
	if !status.Done {
		t.Fatal("Done = false, want true")
	}
	if len(status.Videos) != 1 || status.Videos[0].URL != "https://example.invalid/clip.mp4" {
		t.Errorf("Videos = %+v, want the single clip URL", status.Videos)
	}
	if status.Provider != "acme" {
		t.Errorf("Provider = %q, want acme", status.Provider)
	}
}

func TestPollVideoRequiresJobID(t *testing.T) {
	if _, err := NewOpenAIService("test-key", "https://example.invalid").PollVideo(context.Background(), ""); err == nil {
		t.Fatal("expected error for empty job id")
	}
}
