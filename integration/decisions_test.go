//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/require"
)

func TestOpenRouter_Decisions(t *testing.T) {
	apiKey := keyOrSkip(t, "OPENROUTER_API_KEY")
	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := service.Decide(ctx, aiwire.DecisionOption{
		Model: "typesafe/jev-1.13",
		State: "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
		Questions: map[string]aiwire.DecisionQuestion{
			"department": aiwire.ChoiceQuestion("Which team should handle this", map[string]string{
				"billing":   "Payment or subscription issues",
				"technical": "Bugs or integration problems",
				"sales":     "Pricing or account questions",
			}),
			"frustration": aiwire.ScoreQuestion("How frustrated the customer appears", []string{
				"Calm, just stating facts",
				"Frustrated but civil",
				"Very angry, strong language",
			}),
			"is_urgent": aiwire.NoulQuestion("The message conveys urgency or time-sensitivity"),
		},
	})
	require.NoError(t, err)
	require.Len(t, resp.Answers, 3)

	t.Logf("Provider: %s", resp.Provider)
	for name, a := range resp.Answers {
		t.Logf("%s: %+v", name, a)
	}
	logUsage(t, resp.Usage)

	require.Equal(t, aiwire.DecisionChoice, resp.Answers["department"].Type)
	require.NotEmpty(t, resp.Answers["department"].Choice)
	require.Equal(t, aiwire.DecisionScore, resp.Answers["frustration"].Type)
	require.Equal(t, aiwire.DecisionNoul, resp.Answers["is_urgent"].Type)
	require.Greater(t, resp.Answers["is_urgent"].Noul, 0.5)
}
