package aiwire

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/packages/respjson"
)

func fieldsFromJSON(t *testing.T, raw string) map[string]respjson.Field {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		t.Fatalf("invalid test JSON: %v", err)
	}
	out := make(map[string]respjson.Field, len(obj))
	for k, v := range obj {
		out[k] = respjson.NewField(string(v))
	}
	return out
}

func rawField(t *testing.T, m map[string]json.RawMessage, key string) string {
	t.Helper()
	v, ok := m[key]
	if !ok {
		t.Fatalf("key %q missing from %v", key, m)
	}
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		t.Fatalf("key %q not a string: %v", key, err)
	}
	return s
}

func rawAsMap(t *testing.T, raw json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("raw not an object: %v", err)
	}
	return m
}

func detailFrom(t *testing.T, raw string) ReasoningDetail {
	t.Helper()
	d, ok := reasoningDetailFromRaw(json.RawMessage(raw))
	if !ok {
		t.Fatalf("invalid raw JSON: %s", raw)
	}
	return d
}

func TestExtractReasoningDetails_EncryptedBlock(t *testing.T) {
	fields := fieldsFromJSON(t, `{
		"reasoning_details": [
			{"type":"reasoning.encrypted","data":"opaque-blob","format":"openai-responses-v1","id":"rs_1","index":0}
		]
	}`)

	got := extractReasoningDetails(fields)
	if len(got) != 1 {
		t.Fatalf("expected 1 detail, got %d", len(got))
	}
	d := got[0]
	if d.Type != "reasoning.encrypted" {
		t.Fatalf("type not parsed: %+v", d)
	}
	m := rawAsMap(t, d.Raw)
	if rawField(t, m, "data") != "opaque-blob" || rawField(t, m, "format") != "openai-responses-v1" || rawField(t, m, "id") != "rs_1" {
		t.Fatalf("raw payload missing fields: %s", d.Raw)
	}
}

func TestExtractReasoningDetails_SummaryAndEncryptedMix(t *testing.T) {
	fields := fieldsFromJSON(t, `{
		"reasoning_details": [
			{"type":"reasoning.summary","text":"step one","index":0},
			{"type":"reasoning.encrypted","data":"abc","index":1}
		]
	}`)
	got := extractReasoningDetails(fields)
	if len(got) != 2 {
		t.Fatalf("expected 2, got %d", len(got))
	}
	if rawField(t, rawAsMap(t, got[0].Raw), "text") != "step one" {
		t.Fatalf("idx0 text missing: %s", got[0].Raw)
	}
	if rawField(t, rawAsMap(t, got[1].Raw), "data") != "abc" {
		t.Fatalf("idx1 data missing: %s", got[1].Raw)
	}
}

func TestExtractReasoningDetails_MissingOrEmpty(t *testing.T) {
	cases := map[string]string{
		"missing":  `{"other":"x"}`,
		"null":     `{"reasoning_details": null}`,
		"empty":    `{"reasoning_details": []}`,
		"notArray": `{"reasoning_details": {"foo":"bar"}}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			if got := extractReasoningDetails(fieldsFromJSON(t, raw)); got != nil {
				t.Fatalf("expected nil, got %+v", got)
			}
		})
	}
}

func TestReasoningDetail_MarshalUsesRaw(t *testing.T) {
	d := ReasoningDetail{
		Type: "reasoning.encrypted",
		Raw:  json.RawMessage(`{"type":"reasoning.encrypted","data":"X","unknown_future_field":42}`),
	}
	bytes, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(bytes), "unknown_future_field") {
		t.Fatalf("Raw should round-trip unknown fields; got %s", bytes)
	}
}

func TestReasoningDetail_MarshalFallsBackToTypedFieldsWhenRawEmpty(t *testing.T) {
	d := ReasoningDetail{Type: "reasoning.summary"}
	bytes, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type":"reasoning.summary","index":0}`
	if string(bytes) != want {
		t.Fatalf("got %s want %s", bytes, want)
	}
}

func TestMergeReasoningDetailFragments_AccumulatesByIndex(t *testing.T) {
	var acc reasoningAccum

	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"type":"reasoning.summary","text":"step ","index":0}`),
	})
	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"text":"one","index":0}`),
		detailFrom(t, `{"type":"reasoning.encrypted","data":"abc","index":1}`),
	})
	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"data":"def","index":1}`),
	})

	out := acc.finalize()
	if len(out) != 2 {
		t.Fatalf("expected 2 details, got %d", len(out))
	}
	m0 := rawAsMap(t, out[0].Raw)
	if rawField(t, m0, "text") != "step one" || out[0].Type != "reasoning.summary" {
		t.Fatalf("idx0 not concatenated: %s", out[0].Raw)
	}
	m1 := rawAsMap(t, out[1].Raw)
	if rawField(t, m1, "data") != "def" || out[1].Type != "reasoning.encrypted" {
		t.Fatalf("idx1 should last-write-wins on data: %s", out[1].Raw)
	}
}

func TestMergeReasoningDetailFragments_PreservesSignatureAndUnknowns(t *testing.T) {
	// Anthropic streams the signature on a trailing fragment after the text.
	// Map-based merge should preserve it AND any other unknown wire field.
	var acc reasoningAccum

	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"type":"reasoning.text","text":"thinking ","index":0}`),
	})
	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"text":"more","index":0}`),
	})
	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"signature":"sig-abc123","format":"anthropic-claude-v1","unknown_future":"keep-me","index":0}`),
	})

	out := acc.finalize()
	if len(out) != 1 {
		t.Fatalf("expected 1 detail, got %d", len(out))
	}
	m := rawAsMap(t, out[0].Raw)
	if rawField(t, m, "text") != "thinking more" {
		t.Fatalf("text not concat: %s", out[0].Raw)
	}
	if rawField(t, m, "signature") != "sig-abc123" {
		t.Fatalf("signature lost: %s", out[0].Raw)
	}
	if rawField(t, m, "format") != "anthropic-claude-v1" {
		t.Fatalf("format lost: %s", out[0].Raw)
	}
	if rawField(t, m, "unknown_future") != "keep-me" {
		t.Fatalf("unknown field dropped: %s", out[0].Raw)
	}
}

func TestMergeReasoningDetailFragments_IndexlessFragmentsGetSyntheticSlots(t *testing.T) {
	var acc reasoningAccum

	acc.merge([]ReasoningDetail{
		detailFrom(t, `{"type":"reasoning.text","text":"first"}`),
		detailFrom(t, `{"type":"reasoning.text","text":"second"}`),
	})

	out := acc.finalize()
	if len(out) != 2 {
		t.Fatalf("expected 2 distinct slots, got %d", len(out))
	}
}

func TestAssistantMessageWithReasoning_EmptyDetailsIsPlain(t *testing.T) {
	msg := AssistantMessageWithReasoning("hello", nil, nil)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(bytes), "reasoning_details") {
		t.Fatalf("plain message should not include reasoning_details: %s", bytes)
	}
}

func TestAssistantMessageWithReasoning_AttachesDetails(t *testing.T) {
	details := []ReasoningDetail{
		detailFrom(t, `{"type":"reasoning.encrypted","data":"opaque","format":"openai-responses-v1","id":"rs_1"}`),
	}
	msg := AssistantMessageWithReasoning("done", nil, details)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	s := string(bytes)
	if !strings.Contains(s, `"reasoning_details"`) {
		t.Fatalf("missing reasoning_details: %s", s)
	}
	if !strings.Contains(s, `"opaque"`) || !strings.Contains(s, `"rs_1"`) {
		t.Fatalf("payload missing detail content: %s", s)
	}
	if !strings.Contains(s, `"role":"assistant"`) {
		t.Fatalf("missing assistant role: %s", s)
	}
}

func TestAssistantMessageWithReasoning_RawTakesPrecedence(t *testing.T) {
	details := []ReasoningDetail{{
		Type: "reasoning.encrypted",
		Raw:  json.RawMessage(`{"type":"reasoning.encrypted","data":"X","extra":"keep-me"}`),
	}}
	msg := AssistantMessageWithReasoning("", nil, details)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(bytes), "keep-me") {
		t.Fatalf("raw passthrough lost: %s", bytes)
	}
}
