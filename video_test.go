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
