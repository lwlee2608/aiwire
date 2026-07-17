//go:build integration

package integration

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)

//go:embed testdata/receipt.pdf
var receiptPDF []byte

type receiptLineItem struct {
	Description string  `json:"description" jsonschema:"required"`
	Quantity    int     `json:"quantity" jsonschema:"required"`
	UnitPrice   float64 `json:"unit_price" jsonschema:"required"`
	Amount      float64 `json:"amount" jsonschema:"required"`
}

type receipt struct {
	InvoiceNumber string            `json:"invoice_number" jsonschema:"required"`
	ReceiptNumber string            `json:"receipt_number" jsonschema:"required"`
	DatePaid      string            `json:"date_paid" jsonschema:"required" jsonschema_description:"Date the receipt was paid, as ISO 8601 YYYY-MM-DD"`
	VendorName    string            `json:"vendor_name" jsonschema:"required"`
	BillToName    string            `json:"bill_to_name" jsonschema:"required"`
	BillToEmail   string            `json:"bill_to_email" jsonschema:"required"`
	Currency      string            `json:"currency" jsonschema:"required" jsonschema_description:"ISO 4217 currency code, e.g. USD"`
	Subtotal      float64           `json:"subtotal" jsonschema:"required"`
	Discount      float64           `json:"discount" jsonschema:"required" jsonschema_description:"Total discount applied, as a negative number. 0 if there is no discount."`
	Total         float64           `json:"total" jsonschema:"required"`
	AmountPaid    float64           `json:"amount_paid" jsonschema:"required"`
	PaymentMethod string            `json:"payment_method" jsonschema:"required"`
	LineItems     []receiptLineItem `json:"line_items" jsonschema:"required"`
}

func runReceiptExtraction(t *testing.T, opt aiwire.CompletionOption) {
	t.Helper()

	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	dataURL := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(receiptPDF)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Extract the receipt details from this PDF into the given JSON schema. Use only values present in the document."),
			openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				Filename: openai.String("receipt.pdf"),
				FileData: openai.String(dataURL),
			}),
		}),
	}

	opt.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   "receipt",
				Strict: openai.Bool(true),
				Schema: aiwire.GenerateSchema[receipt](),
			},
		},
	}

	resp, err := service.Completions(context.Background(), messages, nil, opt)
	assert.NoError(t, err)
	if err != nil {
		return
	}
	assert.NotEmpty(t, resp.Message.Content)

	t.Logf("Response: %s", resp.Message.Content)
	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)

	var got receipt
	err = json.Unmarshal([]byte(resp.Message.Content), &got)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	assert.Equal(t, "A1B2C3D4-0001", got.InvoiceNumber)
	assert.Equal(t, "1783-4902-5561", got.ReceiptNumber)
	assert.Equal(t, "2026-06-03", got.DatePaid)
	assert.Contains(t, strings.ToUpper(got.VendorName), "NORTHWIND LABS")
	assert.Contains(t, strings.ToUpper(got.BillToName), "JANE DOE")
	assert.Equal(t, "jane.doe@example.com", got.BillToEmail)
	assert.Equal(t, "USD", got.Currency)
	assert.Equal(t, 22.00, got.Subtotal)
	assert.Equal(t, 11.00, math.Abs(got.Discount))
	assert.Equal(t, 11.00, got.Total)
	assert.Equal(t, 11.00, got.AmountPaid)
	assert.Contains(t, strings.ToUpper(got.PaymentMethod), "VISA")

	// The discount is reported as a line item by some models and via the
	// discount field by others, so only the first row is asserted.
	assert.NotEmpty(t, got.LineItems)
	if len(got.LineItems) > 0 {
		item := got.LineItems[0]
		assert.Contains(t, item.Description, "Creator")
		assert.Equal(t, 1, item.Quantity)
		assert.Equal(t, 22.00, item.UnitPrice)
		assert.Equal(t, 22.00, item.Amount)
	}
}

func TestReadPDF(t *testing.T) {
	models := []string{
		"openai/gpt-5.6-luna",
		"x-ai/grok-4.5",
		"qwen/qwen3.7-plus",
		"moonshotai/kimi-k2.6",
		"google/gemini-3.5-flash",
		"z-ai/glm-4.6v",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			runReceiptExtraction(t, aiwire.CompletionOption{
				Model:       model,
				Temperature: 0.0,
				Provider: &aiwire.ProviderOption{
					AllowFallbacks: true,
				},
			})
		})
	}
}
