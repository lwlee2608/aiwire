package aiwire

import (
	"reflect"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/shared"
)

var timeType = reflect.TypeOf(time.Time{})

func GenerateSchema[T any]() any {
	var v T
	return reflectSchema(reflect.TypeOf(v))
}

func GenerateFunctionParameters[T any]() shared.FunctionParameters {
	var v T
	return reflectSchema(reflect.TypeOf(v))
}

func reflectSchema(t reflect.Type) map[string]any {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil {
		return map[string]any{}
	}

	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": reflectSchema(t.Elem())}
	case reflect.Map, reflect.Interface:
		return map[string]any{"type": "object"}
	case reflect.Struct:
		return reflectStruct(t)
	}
	return map[string]any{}
}

func reflectStruct(t reflect.Type) map[string]any {
	props := map[string]any{}
	var required []string
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name := fieldName(field)
		if name == "" {
			continue
		}
		props[name] = reflectSchema(field.Type)
		if isRequired(field.Tag.Get("jsonschema")) {
			required = append(required, name)
		}
	}
	schema := map[string]any{
		"type":                 "object",
		"properties":           props,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func fieldName(f reflect.StructField) string {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return ""
	}
	if tag == "" {
		return f.Name
	}
	name, _, _ := strings.Cut(tag, ",")
	if name == "" {
		return f.Name
	}
	return name
}

func isRequired(tag string) bool {
	for _, part := range strings.Split(tag, ",") {
		if strings.TrimSpace(part) == "required" {
			return true
		}
	}
	return false
}
