package aiwire

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestGeneratedImageDecode(t *testing.T) {
	raw := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)

	mime, data, err := GeneratedImage{URL: dataURL}.Decode()
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png", mime)
	}
	if string(data) != string(raw) {
		t.Errorf("data = %v, want %v", data, raw)
	}
}

func TestGeneratedImageDecodePlain(t *testing.T) {
	mime, data, err := GeneratedImage{URL: "data:text/plain,hello%20world+a"}.Decode()
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if mime != "text/plain" {
		t.Errorf("mime = %q, want text/plain", mime)
	}
	if string(data) != "hello world+a" {
		t.Errorf("data = %q, want %q", data, "hello world+a")
	}
}

func TestGeneratedImageDecodeErrors(t *testing.T) {
	cases := []string{
		"https://example.com/a.png", // not a data URL
		"data:image/png;base64",     // missing comma
		"data:image/png;base64,!!!", // invalid base64
	}
	for _, url := range cases {
		if _, _, err := (GeneratedImage{URL: url}).Decode(); err == nil {
			t.Errorf("Decode(%q): expected error, got nil", url)
		}
	}
}

func TestImageInputFromBytes(t *testing.T) {
	got := ImageInputFromBytes("image/jpeg", []byte("abc"))
	want := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString([]byte("abc"))
	if got.URL != want {
		t.Errorf("URL = %q, want %q", got.URL, want)
	}
}

func TestImageConfigMap(t *testing.T) {
	if m := imageConfigMap(ImageOption{}); m != nil {
		t.Errorf("empty option: got %v, want nil", m)
	}

	m := imageConfigMap(ImageOption{
		AspectRatio: "16:9",
		ConfigExtra: map[string]any{"seed": 7},
	})
	if m["aspect_ratio"] != "16:9" {
		t.Errorf("aspect_ratio = %v, want 16:9", m["aspect_ratio"])
	}
	if m["seed"] != 7 {
		t.Errorf("seed = %v, want 7", m["seed"])
	}
}

func TestGenerateImageDefaultsToChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if got, want := body["modalities"], []any{"image", "text"}; !reflect.DeepEqual(got, want) {
			t.Errorf("modalities = %#v, want %#v", got, want)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chat-1",
			"object":"chat.completion",
			"created":1,
			"model":"image-model",
			"provider":"Chat Provider",
			"choices":[{
				"index":0,
				"finish_reason":"stop",
				"message":{
					"role":"assistant",
					"content":"caption",
					"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}]
				}
			}],
			"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
		}`))
	}))
	defer server.Close()

	response, err := NewOpenAIService("test-key", server.URL).GenerateImage(context.Background(), ImageOption{
		Model:  "image-model",
		Prompt: "draw something",
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if response.Text != "caption" {
		t.Errorf("Text = %q, want caption", response.Text)
	}
	if response.Provider != "Chat Provider" {
		t.Errorf("Provider = %q, want Chat Provider", response.Provider)
	}
	if len(response.Images) != 1 {
		t.Fatalf("Images length = %d, want 1", len(response.Images))
	}
	if response.Usage.TotalTokens != 3 {
		t.Errorf("TotalTokens = %d, want 3", response.Usage.TotalTokens)
	}
}

func TestGenerateImageViaImagesEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
		if r.URL.Path != "/images" {
			t.Errorf("path = %q, want /images", r.URL.Path)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		want := map[string]any{
			"model":         "openai/gpt-image-2",
			"prompt":        "draw something",
			"aspect_ratio":  "16:9",
			"resolution":    "2K",
			"quality":       "high",
			"output_format": "webp",
			"seed":          float64(7),
			"input_references": []any{
				map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": "data:image/png;base64,YWJj",
					},
				},
			},
		}
		if !reflect.DeepEqual(body, want) {
			t.Errorf("request body = %#v, want %#v", body, want)
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenRouter-Provider", "OpenAI")
		_, _ = w.Write([]byte(`{
			"created":1,
			"data":[
				{"b64_json":"YWJj","media_type":"image/webp"},
				{"url":"https://example.com/generated.png"}
			],
			"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9,"cost":0.04}
		}`))
	}))
	defer server.Close()

	response, err := NewOpenAIService("test-key", server.URL).GenerateImage(context.Background(), ImageOption{
		Model:        "openai/gpt-image-2",
		Prompt:       "draw something",
		Endpoint:     ImageEndpointImages,
		Images:       []ImageInput{{URL: "data:image/png;base64,YWJj"}},
		AspectRatio:  "16:9",
		Resolution:   "2K",
		Quality:      "high",
		OutputFormat: "webp",
		ConfigExtra:  map[string]any{"seed": 7},
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if response.Provider != "OpenAI" {
		t.Errorf("Provider = %q, want OpenAI", response.Provider)
	}
	if len(response.Images) != 2 {
		t.Fatalf("Images length = %d, want 2", len(response.Images))
	}
	if got, want := response.Images[0].URL, "data:image/webp;base64,YWJj"; got != want {
		t.Errorf("first image URL = %q, want %q", got, want)
	}
	if got, want := response.Images[1].URL, "https://example.com/generated.png"; got != want {
		t.Errorf("second image URL = %q, want %q", got, want)
	}
	if response.Usage.PromptTokens != 4 || response.Usage.CompletionTokens != 5 || response.Usage.TotalTokens != 9 {
		t.Errorf("Usage = %+v, want prompt=4 completion=5 total=9", response.Usage)
	}
	if response.Usage.Cost != 0.04 {
		t.Errorf("Cost = %f, want 0.04", response.Usage.Cost)
	}
}

func TestGenerateImageViaImagesEndpointReturnsAPIErrorWithoutRetry(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/images" {
			t.Errorf("path = %q, want /images", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid image request","code":400}}`))
	}))
	defer server.Close()

	_, err := NewOpenAIService("test-key", server.URL).GenerateImage(context.Background(), ImageOption{
		Model:    "openai/gpt-image-2",
		Prompt:   "draw something",
		Endpoint: ImageEndpointImages,
	})
	if err == nil {
		t.Fatal("GenerateImage: expected error, got nil")
	}
	if requests != 1 {
		t.Errorf("requests = %d, want 1", requests)
	}
}

func TestGenerateImageRejectsUnknownEndpoint(t *testing.T) {
	_, err := NewOpenAIService("test-key", "https://example.com").GenerateImage(context.Background(), ImageOption{
		Endpoint: ImageEndpoint("unknown"),
	})
	if err == nil {
		t.Fatal("GenerateImage: expected error, got nil")
	}
}
