package aiwire

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/respjson"
)

// ImageGeneration generates images from a text prompt and optional input
// images. It is intentionally separate from [Completion]: image generation may
// be served by endpoints other than chat/completions in future (OpenAI Images,
// Flux, and so on), so it does not share CompletionOption / CompletionResponse.
type ImageGeneration interface {
	GenerateImage(ctx context.Context, opt ImageOption) (ImageResponse, error)
}

// ImageOption configures an image-generation request.
type ImageOption struct {
	Model       string
	Prompt      string
	Images      []ImageInput   // optional source images for editing or reference
	AspectRatio string         // e.g. "16:9"; forwarded inside image_config
	ConfigExtra map[string]any // extra image_config knobs, merged as-is
	// Modalities sets the requested output modalities; defaults to
	// ["image", "text"]. Use ["image"] for image-only models that reject text.
	Modalities []string
	// Provider holds OpenRouter-style routing; nil for other backends.
	Provider *ProviderOption
}

// ImageInput is a source image supplied for editing or as a reference.
type ImageInput struct {
	URL string // data URL ("data:image/png;base64,...") or a remote URL
}

// ImageInputFromBytes builds an ImageInput as a base64 data URL.
func ImageInputFromBytes(mimeType string, data []byte) ImageInput {
	return ImageInput{
		URL: "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data),
	}
}

// ImageResponse is the result of an image-generation request.
type ImageResponse struct {
	Images   []GeneratedImage
	Text     string // accompanying assistant text, if the model emits any
	Provider string
	Usage    Usage
}

// GeneratedImage is one image emitted by an image-generation model.
type GeneratedImage struct {
	Type string // "image_url"
	URL  string // "data:image/png;base64,..." or a remote URL
}

// Decode parses a data: URL into its MIME type and raw bytes. It returns an
// error for remote (non-data) URLs or malformed data; it never panics.
func (g GeneratedImage) Decode() (mimeType string, data []byte, err error) {
	rest, ok := strings.CutPrefix(g.URL, "data:")
	if !ok {
		return "", nil, fmt.Errorf("aiwire: not a data URL: %q", g.URL)
	}
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "", nil, errors.New("aiwire: malformed data URL: missing comma")
	}
	if mt, isBase64 := strings.CutSuffix(meta, ";base64"); isBase64 {
		decoded, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return "", nil, fmt.Errorf("aiwire: decode base64 data URL: %w", err)
		}
		return mt, decoded, nil
	}
	// Plain (percent-encoded) data URL — uncommon for images, supported for completeness.
	decoded, err := url.PathUnescape(payload)
	if err != nil {
		return "", nil, fmt.Errorf("aiwire: decode data URL: %w", err)
	}
	return meta, []byte(decoded), nil
}

// GenerateImage implements [ImageGeneration] over an OpenAI-compatible
// chat/completions endpoint (e.g. OpenRouter). It requests image output via
// modalities and reads the generated images back from the assistant message.
func (s *Service) GenerateImage(ctx context.Context, opt ImageOption) (ImageResponse, error) {
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, 1+len(opt.Images))
	if opt.Prompt != "" {
		parts = append(parts, openai.TextContentPart(opt.Prompt))
	}
	for _, img := range opt.Images {
		parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
			URL: img.URL,
		}))
	}

	modalities := opt.Modalities
	if len(modalities) == 0 {
		modalities = []string{"image", "text"}
	}

	params := openai.ChatCompletionNewParams{
		Model:      opt.Model,
		Messages:   []openai.ChatCompletionMessageParamUnion{openai.UserMessage(parts)},
		Modalities: modalities,
	}

	reqOpts := buildRequestOptions(opt.Provider, nil)
	if cfg := imageConfigMap(opt); cfg != nil {
		reqOpts = append(reqOpts, option.WithJSONSet("image_config", cfg))
	}

	var response *http.Response
	reqOpts = append(reqOpts, option.WithResponseInto(&response))

	completion, err := s.client.Chat.Completions.New(ctx, params, reqOpts...)
	if err != nil {
		return ImageResponse{}, err
	}
	if len(completion.Choices) == 0 {
		return ImageResponse{}, errors.New("no choices returned from image generation")
	}

	message := completion.Choices[0].Message

	return ImageResponse{
		Images:   extractImages(message.JSON.ExtraFields),
		Text:     message.Content,
		Provider: resolveRoutedProvider(completion.JSON.ExtraFields, message.JSON.ExtraFields, response),
		Usage:    UsageFromOpenAI(completion.Usage),
	}, nil
}

func imageConfigMap(opt ImageOption) map[string]any {
	m := map[string]any{}
	if opt.AspectRatio != "" {
		m["aspect_ratio"] = opt.AspectRatio
	}
	for k, v := range opt.ConfigExtra {
		m[k] = v
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// extractImages pulls the OpenRouter `images` array out of an assistant
// message's extra fields. Each entry is {type, image_url:{url}}. Malformed
// entries are skipped; nil is returned when the field is absent or empty.
func extractImages(extraFields map[string]respjson.Field) []GeneratedImage {
	field, ok := extraFields["images"]
	if !ok {
		return nil
	}
	raw := strings.TrimSpace(field.Raw())
	if raw == "" || raw == "null" {
		return nil
	}

	var items []struct {
		Type     string `json:"type"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal([]byte(raw), &items); err != nil {
		return nil
	}

	out := make([]GeneratedImage, 0, len(items))
	for _, it := range items {
		if it.ImageURL.URL == "" {
			continue
		}
		typ := it.Type
		if typ == "" {
			typ = "image_url"
		}
		out = append(out, GeneratedImage{Type: typ, URL: it.ImageURL.URL})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
