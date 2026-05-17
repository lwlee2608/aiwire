package aiwire

import (
	"context"
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

func buildResponsesParams(input responses.ResponseInputParam, tools []responses.ToolUnionParam, opt ResponsesOption) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(opt.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Text:  responses.ResponseTextConfigParam{Format: opt.ResponseFormat},
	}

	if opt.Temperature != 0 {
		params.Temperature = openai.Float(opt.Temperature)
	}
	if opt.MaxOutputTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*opt.MaxOutputTokens))
	}
	if opt.Instructions != "" {
		params.Instructions = openai.String(opt.Instructions)
	}
	if opt.PreviousResponseID != "" {
		params.PreviousResponseID = openai.String(opt.PreviousResponseID)
	}
	if opt.Store != nil {
		params.Store = openai.Bool(*opt.Store)
	}
	if len(opt.Include) > 0 {
		inc := make([]responses.ResponseIncludable, len(opt.Include))
		for i, s := range opt.Include {
			inc[i] = responses.ResponseIncludable(s)
		}
		params.Include = inc
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	return params
}

func (s *Service) Respond(
	ctx context.Context,
	input responses.ResponseInputParam,
	tools []responses.ToolUnionParam,
	opt ResponsesOption,
) (ResponsesResponse, error) {
	params := buildResponsesParams(input, tools, opt)

	var httpResp *http.Response
	opts := buildRequestOptions(opt.Provider, opt.Reasoning)
	opts = append(opts, option.WithResponseInto(&httpResp))

	resp, err := s.client.Responses.New(ctx, params, opts...)
	if err != nil {
		return ResponsesResponse{}, err
	}

	provider := extractStringFromExtraFields(resp.JSON.ExtraFields, "provider", "provider_name")
	if provider == "" {
		provider = extractProviderFromHeader(httpResp)
	}

	return ResponsesResponse{
		ID:       resp.ID,
		Status:   string(resp.Status),
		Output:   resp.Output,
		Provider: provider,
		Usage:    UsageFromResponses(resp.Usage),
	}, nil
}

func (s *Service) RespondStream(
	ctx context.Context,
	input responses.ResponseInputParam,
	tools []responses.ToolUnionParam,
	opt ResponsesOption,
	callback ResponsesStreamCallback,
) error {
	params := buildResponsesParams(input, tools, opt)

	var httpResp *http.Response
	opts := buildRequestOptions(opt.Provider, opt.Reasoning)
	opts = append(opts, option.WithResponseInto(&httpResp))

	stream := s.client.Responses.NewStreaming(ctx, params, opts...)
	defer stream.Close()

	var (
		responseID     string
		finalUsage     Usage
		hasFinalUsage  bool
		routedProvider string
		headerChecked  bool
	)

	for stream.Next() {
		ev := stream.Current()

		if !headerChecked {
			routedProvider = extractProviderFromHeader(httpResp)
			headerChecked = true
		}

		if ev.Response.ID != "" {
			responseID = ev.Response.ID
		}

		if p := extractStringFromExtraFields(ev.Response.JSON.ExtraFields, "provider", "provider_name"); p != "" {
			routedProvider = p
		}

		chunk := ResponsesStreamChunk{
			Type:       ev.Type,
			Delta:      ev.Delta,
			ResponseID: responseID,
			Provider:   routedProvider,
		}

		// `added` carries an empty shell; only `done` has the completed item.
		if ev.Type == "response.output_item.done" {
			item := ev.Item
			chunk.Item = &item
		}

		if ev.Type == "response.completed" || ev.Type == "response.incomplete" || ev.Type == "response.failed" {
			if ev.Response.JSON.Usage.Valid() {
				finalUsage = UsageFromResponses(ev.Response.Usage)
				hasFinalUsage = true
			}
			chunk.FinishReason = string(ev.Response.Status)
		}

		if err := callback(chunk); err != nil {
			return err
		}
	}

	if err := stream.Err(); err != nil {
		return err
	}

	finalChunk := ResponsesStreamChunk{
		Done:       true,
		ResponseID: responseID,
		Provider:   routedProvider,
	}
	if hasFinalUsage {
		finalChunk.Usage = &finalUsage
	}
	return callback(finalChunk)
}
