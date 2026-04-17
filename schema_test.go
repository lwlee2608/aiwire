package aiwire

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReflectSchema_Primitives(t *testing.T) {
	cases := []struct {
		name string
		in   reflect.Type
		want map[string]any
	}{
		{"string", reflect.TypeOf(""), map[string]any{"type": "string"}},
		{"bool", reflect.TypeOf(true), map[string]any{"type": "boolean"}},
		{"int", reflect.TypeOf(int(0)), map[string]any{"type": "integer"}},
		{"int64", reflect.TypeOf(int64(0)), map[string]any{"type": "integer"}},
		{"uint32", reflect.TypeOf(uint32(0)), map[string]any{"type": "integer"}},
		{"float64", reflect.TypeOf(float64(0)), map[string]any{"type": "number"}},
		{"float32", reflect.TypeOf(float32(0)), map[string]any{"type": "number"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, reflectSchema(tc.in))
		})
	}
}

func TestReflectSchema_Slice(t *testing.T) {
	got := reflectSchema(reflect.TypeOf([]string{}))
	assert.Equal(t, map[string]any{
		"type":  "array",
		"items": map[string]any{"type": "string"},
	}, got)
}

func TestReflectSchema_Pointer(t *testing.T) {
	var p *int
	got := reflectSchema(reflect.TypeOf(p))
	assert.Equal(t, map[string]any{"type": "integer"}, got)
}

func TestReflectSchema_MapAndInterface(t *testing.T) {
	assert.Equal(t, map[string]any{"type": "object"}, reflectSchema(reflect.TypeOf(map[string]int{})))
	var iface any
	assert.Equal(t, map[string]any{"type": "object"}, reflectSchema(reflect.TypeOf(&iface).Elem()))
}

func TestReflectSchema_Struct_BasicRequiredAndTags(t *testing.T) {
	type inner struct {
		Name    string `json:"name" jsonschema:"required"`
		Age     int    `json:"age"`
		Ignored string `json:"-"`
		unexp   string //nolint:unused
	}
	got := reflectSchema(reflect.TypeOf(inner{}))
	assert.Equal(t, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required":             []string{"name"},
		"additionalProperties": false,
	}, got)
}

func TestReflectSchema_Struct_NoJSONTagUsesFieldName(t *testing.T) {
	type s struct {
		Value string
	}
	got := reflectSchema(reflect.TypeOf(s{}))
	props := got["properties"].(map[string]any)
	_, ok := props["Value"]
	assert.True(t, ok, "expected property keyed by Go field name when json tag is absent")
}

func TestReflectSchema_Struct_Nested(t *testing.T) {
	type inner struct {
		X int `json:"x" jsonschema:"required"`
	}
	type outer struct {
		Child inner `json:"child" jsonschema:"required"`
	}
	got := reflectSchema(reflect.TypeOf(outer{}))
	child := got["properties"].(map[string]any)["child"].(map[string]any)
	assert.Equal(t, "object", child["type"])
	assert.Equal(t, []string{"x"}, child["required"])
}

func TestGenerateFunctionParameters(t *testing.T) {
	type input struct {
		A int `json:"a" jsonschema:"required"`
	}
	got := GenerateFunctionParameters[input]()
	assert.Equal(t, "object", got["type"])
	assert.Equal(t, []string{"a"}, got["required"])
}

func TestGenerateSchema(t *testing.T) {
	type input struct {
		A int `json:"a" jsonschema:"required"`
	}
	got, ok := GenerateSchema[input]().(map[string]any)
	assert.True(t, ok)
	assert.Equal(t, "object", got["type"])
}
