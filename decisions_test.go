package aiwire

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDecideSendsQuestionsAndParsesAnswers(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"dec-1","model":"typesafe/jev-1.13","provider":"TypeSafe",
			"answers":{
				"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.84,"technical":0.16},"confidence":0.6},
				"frustration":{"type":"score","score":1.03,"legend":{"0":"calm","1":"annoyed"},"confidence":0.84},
				"is_urgent":{"type":"noul","noul":0.999}
			},
			"usage":{"input_tokens":312,"output_tokens":48,"cost":0.00001}
		}`))
	}))
	defer server.Close()

	resp, err := NewOpenAIService("test-key", server.URL+"/api/v1").Decide(context.Background(), DecisionOption{
		Model: "typesafe/jev-1.13",
		State: "Stripe keeps failing, losing sales, help ASAP",
		Questions: map[string]DecisionQuestion{
			"department":  ChoiceQuestion("Which team", map[string]string{"billing": "payments", "technical": "bugs"}),
			"frustration": ScoreQuestion("How frustrated", []string{"calm", "annoyed"}),
			"is_urgent":   NoulQuestion("Is it urgent"),
		},
		Provider:  &ProviderOption{AllowFallbacks: true},
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}

	if gotPath != "/api/alpha/decisions" {
		t.Errorf("path = %q, want /api/alpha/decisions", gotPath)
	}
	if gotBody["model"] != "typesafe/jev-1.13" || gotBody["session_id"] != "sess-1" {
		t.Errorf("body = %v", gotBody)
	}
	if _, ok := gotBody["provider"]; !ok {
		t.Error("provider preferences not sent")
	}
	questions := gotBody["questions"].(map[string]any)
	if q := questions["department"].(map[string]any); q["type"] != "choice" || q["criteria"].(map[string]any)["billing"] != "payments" {
		t.Errorf("choice question = %v", q)
	}
	if q := questions["is_urgent"].(map[string]any); q["type"] != "noul" {
		t.Errorf("noul question = %v", q)
	} else if _, ok := q["criteria"]; ok {
		t.Error("noul question should omit criteria")
	}

	if resp.ID != "dec-1" || resp.Provider != "TypeSafe" {
		t.Errorf("resp = %+v", resp)
	}
	if a := resp.Answers["department"]; a.Type != DecisionChoice || a.Choice != "billing" || a.Probabilities["billing"] != 0.84 || a.Confidence != 0.6 {
		t.Errorf("choice answer = %+v", a)
	}
	if a := resp.Answers["frustration"]; a.Type != DecisionScore || a.Score != 1.03 || a.Legend["1"] != "annoyed" {
		t.Errorf("score answer = %+v", a)
	}
	if a := resp.Answers["is_urgent"]; a.Type != DecisionNoul || a.Noul != 0.999 {
		t.Errorf("noul answer = %+v", a)
	}
	if resp.Usage.PromptTokens != 312 || resp.Usage.CompletionTokens != 48 || resp.Usage.TotalTokens != 360 || resp.Usage.Cost != 0.00001 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestDecideFallsBackToProviderHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenRouter-Provider", "TypeSafe")
		_, _ = w.Write([]byte(`{"id":"dec-1","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	resp, err := NewOpenAIService("test-key", server.URL).Decide(context.Background(), DecisionOption{
		Model:     "typesafe/jev-1.13",
		State:     "x",
		Questions: map[string]DecisionQuestion{"q": NoulQuestion("?")},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if resp.Provider != "TypeSafe" {
		t.Errorf("provider = %q, want TypeSafe from header", resp.Provider)
	}
}

func TestDecideRejectsMissingAnswer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","answers":{"present":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	resp, err := NewOpenAIService("test-key", server.URL+"/api/v1").Decide(context.Background(), DecisionOption{
		Model: "m",
		State: "x",
		Questions: map[string]DecisionQuestion{
			"present": NoulQuestion("Present?"),
			"missing": NoulQuestion("Missing?"),
		},
	})
	require.EqualError(t, err, `aiwire: decisions response missing answer for question "missing"`)
	require.Equal(t, DecisionResponse{}, resp)
}

func TestDecideRequiresQuestions(t *testing.T) {
	_, err := NewOpenAIService("test-key", "http://127.0.0.1:0").Decide(context.Background(), DecisionOption{Model: "m", State: "x"})
	if err == nil || !strings.Contains(err.Error(), "no questions") {
		t.Fatalf("error = %v, want missing questions error", err)
	}
}
