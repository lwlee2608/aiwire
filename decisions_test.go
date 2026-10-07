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

func TestDecisionAnswerProbability(t *testing.T) {
	choice := DecisionAnswer{
		Type:          DecisionChoice,
		Choice:        "block",
		Probabilities: map[string]float64{"block": 0.73, "comment": 0.26, "approve": 0.01},
	}
	require.Equal(t, 0.73, choice.Probability("block"))
	require.Equal(t, 0.0, choice.Probability("unknown"))

	score := DecisionAnswer{
		Type:          DecisionScore,
		Score:         2.5,
		Legend:        map[string]string{"0": "none", "1": "minor", "2": "major", "3": "critical"},
		Probabilities: map[string]float64{"0": 0.02, "1": 0.04, "2": 0.36, "3": 0.58},
	}
	require.Equal(t, 0.58, score.Probability("critical"))
	require.Equal(t, 0.0, score.Probability("3"))
	require.Equal(t, 0.0, score.Probability("unknown"))

	numericScore := DecisionAnswer{
		Type:          DecisionScore,
		Legend:        map[string]string{"0": "1", "1": "2", "2": "3"},
		Probabilities: map[string]float64{"0": 0.1, "1": 0.2, "2": 0.7},
	}
	require.Equal(t, 0.1, numericScore.Probability("1"))
	require.Equal(t, 0.7, numericScore.Probability("3"))
	require.Equal(t, 0.0, numericScore.Probability("0"))

	scoreWithoutLegend := DecisionAnswer{
		Type:          DecisionScore,
		Probabilities: map[string]float64{"0": 0.2, "1": 0.8},
	}
	require.Equal(t, 0.8, scoreWithoutLegend.Probability("1"))
	require.Equal(t, 0.0, scoreWithoutLegend.Probability("high"))

	noul := DecisionAnswer{Type: DecisionNoul, Noul: 0.86}
	require.Equal(t, 0.0, noul.Probability("yes"))
}
