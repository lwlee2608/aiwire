//go:build integration

package integration

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

const ocrImageURL = "https://placehold.co/400x100/000000/FFFFFF/png?text=OCR+TEST+42"

const ocrPDFText = "AIWIRE OCR 7391"

// ocrBase64PNG is a 300x80 PNG showing "BASE64 OCR 99" in white on black.
//
//go:embed testdata/ocr_base64.png
var ocrBase64PNG []byte

func runOCRImage(t *testing.T, service *aiwire.Service, opt aiwire.CompletionOption) {
	t.Helper()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Read the text in this image. Reply with only the text, no commentary."),
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: ocrImageURL,
			}),
		}),
	}

	resp, err := service.Completions(context.Background(), messages, nil, opt)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp.Message.Content)

	t.Logf("Image OCR response: %s", resp.Message.Content)
	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)

	got := strings.ToUpper(resp.Message.Content)
	assert.Contains(t, got, "OCR")
	assert.Contains(t, got, "TEST")
	assert.Contains(t, got, "42")
}

func runOCRImageBase64(t *testing.T, service *aiwire.Service, opt aiwire.CompletionOption) {
	t.Helper()
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(ocrBase64PNG)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Read the text in this image. Reply with only the text, no commentary."),
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: dataURL,
			}),
		}),
	}

	resp, err := service.Completions(context.Background(), messages, nil, opt)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp.Message.Content)

	t.Logf("Image (base64) OCR response: %s", resp.Message.Content)
	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)

	got := strings.ToUpper(resp.Message.Content)
	assert.Contains(t, got, "BASE64")
	assert.Contains(t, got, "OCR")
	assert.Contains(t, got, "99")
}

func runOCRPDF(t *testing.T, service *aiwire.Service, opt aiwire.CompletionOption) {
	t.Helper()
	pdfBytes := buildTinyPDF(ocrPDFText)
	dataURL := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfBytes)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Read the text in this PDF. Reply with only the text, no commentary."),
			openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				Filename: openai.String("ocr.pdf"),
				FileData: openai.String(dataURL),
			}),
		}),
	}

	resp, err := service.Completions(context.Background(), messages, nil, opt)
	assert.NoError(t, err)
	assert.NotEmpty(t, resp.Message.Content)

	t.Logf("PDF OCR response: %s", resp.Message.Content)
	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)

	got := strings.ToUpper(resp.Message.Content)
	for token := range strings.FieldsSeq(ocrPDFText) {
		assert.Contains(t, got, strings.ToUpper(token))
	}
}

func TestOpenAI_OCR_Image(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	runOCRImage(t, service, aiwire.CompletionOption{
		Model:       "gpt-4.1-mini",
		Temperature: 0.0,
	})
}

func TestOpenAI_OCR_ImageBase64(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	runOCRImageBase64(t, service, aiwire.CompletionOption{
		Model:       "gpt-4.1-mini",
		Temperature: 0.0,
	})
}

func TestOpenAI_OCR_PDF(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	runOCRPDF(t, service, aiwire.CompletionOption{
		Model:       "gpt-4.1-mini",
		Temperature: 0.0,
	})
}

func TestOpenRouter_OCR_Image(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	runOCRImage(t, service, aiwire.CompletionOption{
		Model:       "z-ai/glm-5v-turbo",
		Temperature: 0.0,
	})
}

func TestOpenRouter_OCR_ImageBase64(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	runOCRImageBase64(t, service, aiwire.CompletionOption{
		Model:       "z-ai/glm-5v-turbo",
		Temperature: 0.0,
	})
}

func TestOpenRouter_OCR_PDF(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
	runOCRPDF(t, service, aiwire.CompletionOption{
		Model:       "z-ai/glm-5v-turbo",
		Temperature: 0.0,
	})
}

// buildTinyPDF returns a minimal single-page PDF that prints text in Helvetica
// at 36pt. Good enough for vision models to OCR; not a general-purpose writer.
func buildTinyPDF(text string) []byte {
	var buf bytes.Buffer
	offsets := make([]int, 6)

	buf.WriteString("%PDF-1.4\n")

	offsets[1] = buf.Len()
	buf.WriteString("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

	offsets[2] = buf.Len()
	buf.WriteString("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

	offsets[3] = buf.Len()
	buf.WriteString("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]" +
		" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")

	stream := fmt.Sprintf("BT /F1 36 Tf 72 700 Td (%s) Tj ET", text)
	offsets[4] = buf.Len()
	fmt.Fprintf(&buf, "4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n", len(stream), stream)

	offsets[5] = buf.Len()
	buf.WriteString("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

	xref := buf.Len()
	buf.WriteString("xref\n0 6\n0000000000 65535 f \n")
	for i := 1; i <= 5; i++ {
		fmt.Fprintf(&buf, "%010d 00000 n \n", offsets[i])
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", xref)

	return buf.Bytes()
}
