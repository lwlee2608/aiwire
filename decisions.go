package aiwire

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3/option"
)

var _ Decisions = (*Service)(nil)

// Decisions asks a decision model (e.g. typesafe/jev-1.13) typed questions
// about a piece of state and returns typed answers rather than text. It is
// separate from [Completion] because these models do not speak chat
// completions: OpenRouter exposes them on its alpha /decisions endpoint.
type Decisions interface {
	Decide(ctx context.Context, opt DecisionOption) (DecisionResponse, error)
}

type DecisionQuestionType string

const (
	DecisionNoul   DecisionQuestionType = "noul"
	DecisionChoice DecisionQuestionType = "choice"
	DecisionScore  DecisionQuestionType = "score"
)

// DecisionOption configures a decisions request.
type DecisionOption struct {
	Model     string
	State     any // plain string, or a JSON object/array of related context
	Questions map[string]DecisionQuestion
	Provider  *ProviderOption
	SessionID string
}

// DecisionQuestion is one question keyed by name in [DecisionOption.Questions].
// Build it with [NoulQuestion], [ChoiceQuestion], or [ScoreQuestion].
type DecisionQuestion struct {
	Type         DecisionQuestionType `json:"type"`
	Instructions string               `json:"instructions"`
	Criteria     any                  `json:"criteria,omitempty"`
}

// NoulQuestion asks a yes/no question; the answer is the probability of yes.
func NoulQuestion(instructions string) DecisionQuestion {
	return DecisionQuestion{Type: DecisionNoul, Instructions: instructions}
}

// ChoiceQuestion picks one option; options maps option name to its description.
func ChoiceQuestion(instructions string, options map[string]string) DecisionQuestion {
	return DecisionQuestion{Type: DecisionChoice, Instructions: instructions, Criteria: options}
}

// ScoreQuestion rates the state against ordered levels; the answer is an index
// into levels (fractional between adjacent levels).
func ScoreQuestion(instructions string, levels []string) DecisionQuestion {
	return DecisionQuestion{Type: DecisionScore, Instructions: instructions, Criteria: levels}
}

type DecisionResponse struct {
	ID       string
	Model    string
	Provider string
	Answers  map[string]DecisionAnswer
	Usage    Usage
}

// DecisionAnswer is a typed answer; Type selects which of Noul, Choice, or
// Score is populated. Probabilities and Confidence are absent for noul.
type DecisionAnswer struct {
	Type          DecisionQuestionType `json:"type"`
	Noul          float64              `json:"noul"`
	Choice        string               `json:"choice"`
	Score         float64              `json:"score"`
	Legend        map[string]string    `json:"legend,omitempty"`
	Probabilities map[string]float64   `json:"probabilities,omitempty"`
	Confidence    float64              `json:"confidence"`
}

func (s *Service) Decide(ctx context.Context, opt DecisionOption) (DecisionResponse, error) {
	if len(opt.Questions) == 0 {
		return DecisionResponse{}, errors.New("aiwire: decisions request has no questions")
	}

	body := map[string]any{
		"model":     opt.Model,
		"state":     opt.State,
		"questions": opt.Questions,
	}
	if opt.SessionID != "" {
		body["session_id"] = opt.SessionID
	}

	var response *http.Response
	var result struct {
		ID       string                    `json:"id"`
		Model    string                    `json:"model"`
		Provider string                    `json:"provider"`
		Answers  map[string]DecisionAnswer `json:"answers"`
		Usage    struct {
			InputTokens  int64   `json:"input_tokens"`
			OutputTokens int64   `json:"output_tokens"`
			Cost         float64 `json:"cost"`
		} `json:"usage"`
	}
	// The decisions endpoint lives at /api/alpha, a sibling of the /api/v1
	// base URL, so step out of v1 rather than hard-coding the host.
	opts := append(buildRequestOptions(opt.Provider, nil), option.WithResponseInto(&response))
	if err := s.client.Post(ctx, "../alpha/decisions", body, &result, opts...); err != nil {
		return DecisionResponse{}, err
	}

	provider := strings.TrimSpace(result.Provider)
	if provider == "" {
		provider = extractProviderFromHeader(response)
	}

	return DecisionResponse{
		ID:       result.ID,
		Model:    result.Model,
		Provider: provider,
		Answers:  result.Answers,
		Usage: Usage{
			PromptTokens:     result.Usage.InputTokens,
			CompletionTokens: result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
			Cost:             result.Usage.Cost,
		},
	}, nil
}
