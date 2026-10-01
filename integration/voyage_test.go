//go:build integration

package integration

import (
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoyage_EmbeddingInputType(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "VOYAGE_API_KEY"), "https://api.voyageai.com/v1")
	text := []string{"Paris is the capital of France."}

	plain, err := service.EmbeddingBatch(t.Context(), text, "voyage-4-large")
	require.NoError(t, err)
	doc, err := service.EmbeddingBatch(t.Context(), text, "voyage-4-large", aiwire.EmbeddingOption{InputType: aiwire.EmbeddingInputDocument})
	require.NoError(t, err)
	query, err := service.Embedding(t.Context(), text[0], "voyage-4-large", aiwire.EmbeddingOption{InputType: aiwire.EmbeddingInputQuery})
	require.NoError(t, err)

	assert.Equal(t, 1024, len(doc[0]))
	assert.NotEqual(t, plain[0], doc[0])
	assert.NotEqual(t, doc[0], query)
}
