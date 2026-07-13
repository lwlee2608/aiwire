package aiwire

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
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

// ImageEndpoint selects the API used for image generation.
type ImageEndpoint string

const (
	ImageEndpointChatCompletions ImageEndpoint = "chat_completions"
	ImageEndpointImages          ImageEndpoint = "images"
)

// ImageOption configures an image-generation request.
type ImageOption struct {
	Model        string
	Prompt       string
	Images       []ImageInput    // optional source images for editing or reference
	Endpoint     ImageEndpoint   // defaults to chat_completions
	AspectRatio  string          // e.g. "16:9"
	Resolution   string          // e.g. "2K"; used by the images endpoint
	Quality      string          // e.g. "high"; used by the images endpoint
	OutputFormat string          // e.g. "png"; used by the images endpoint
	ConfigExtra  map[string]any  // extra image_config or top-level images endpoint knobs
	Modalities   []string        // defaults to ["image", "text"]; use ["image"] for image-only models
	Provider     *ProviderOption // optional OpenRouter-style routing for chat_completions only
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

// GenerateImage implements [ImageGeneration]. The zero-value endpoint uses
// chat/completions for compatibility; ImageEndpointImages uses OpenRouter's
// dedicated /images API.
func (s *Service) GenerateImage(ctx context.Context, opt ImageOption) (ImageResponse, error) {
	switch opt.Endpoint {
	case "", ImageEndpointChatCompletions:
		return s.generateImageViaChatCompletions(ctx, opt)
	case ImageEndpointImages:
		return s.generateImageViaImages(ctx, opt)
	default:
		return ImageResponse{}, fmt.Errorf("aiwire: unsupported image endpoint %q", opt.Endpoint)
	}
}

func (s *Service) generateImageViaChatCompletions(ctx context.Context, opt ImageOption) (ImageResponse, error) {
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

func (s *Service) generateImageViaImages(ctx context.Context, opt ImageOption) (ImageResponse, error) {
	if opt.Provider != nil {
		return ImageResponse{}, errors.New("aiwire: ImageOption.Provider is unsupported by the images endpoint; use ConfigExtra[\"provider\"] for Images API provider options")
	}

	body := map[string]any{
		"model":  opt.Model,
		"prompt": opt.Prompt,
	}
	if len(opt.Images) > 0 {
		references := make([]map[string]any, 0, len(opt.Images))
		for _, image := range opt.Images {
			references = append(references, map[string]any{
				"type": "image_url",
				"image_url": map[string]string{
					"url": image.URL,
				},
			})
		}
		body["input_references"] = references
	}

	setImageParameter(body, "aspect_ratio", opt.AspectRatio)
	setImageParameter(body, "resolution", opt.Resolution)
	setImageParameter(body, "quality", opt.Quality)
	setImageParameter(body, "output_format", opt.OutputFormat)
	maps.Copy(body, opt.ConfigExtra)

	var response *http.Response

	var result struct {
		Data []struct {
			B64JSON   string `json:"b64_json"`
			MediaType string `json:"media_type"`
			URL       string `json:"url"`
		} `json:"data"`
		Provider     string                 `json:"provider"`
		ProviderName string                 `json:"provider_name"`
		Usage        openai.CompletionUsage `json:"usage"`
	}
	if err := s.client.Post(ctx, "images", body, &result, option.WithResponseInto(&response)); err != nil {
		return ImageResponse{}, err
	}

	images := make([]GeneratedImage, 0, len(result.Data))
	for _, image := range result.Data {
		imageURL := image.URL
		if imageURL == "" && image.B64JSON != "" {
			mediaType := image.MediaType
			if mediaType == "" {
				mediaType = "application/octet-stream"
			}
			imageURL = "data:" + mediaType + ";base64," + image.B64JSON
		}
		if imageURL != "" {
			images = append(images, GeneratedImage{Type: "image_url", URL: imageURL})
		}
	}

	provider := strings.TrimSpace(result.Provider)
	if provider == "" {
		provider = strings.TrimSpace(result.ProviderName)
	}
	if provider == "" {
		provider = extractProviderFromHeader(response)
	}

	return ImageResponse{
		Images:   images,
		Provider: provider,
		Usage:    UsageFromOpenAI(result.Usage),
	}, nil
}

func setImageParameter(body map[string]any, key, value string) {
	if value != "" {
		body[key] = value
	}
}

func imageConfigMap(opt ImageOption) map[string]any {
	m := map[string]any{}
	if opt.AspectRatio != "" {
		m["aspect_ratio"] = opt.AspectRatio
	}
	maps.Copy(m, opt.ConfigExtra)
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
