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
	return reflectSchema(reflect.TypeOf(v), map[reflect.Type]bool{})
}

func GenerateFunctionParameters[T any]() shared.FunctionParameters {
	var v T
	return reflectSchema(reflect.TypeOf(v), map[reflect.Type]bool{})
}

func reflectSchema(t reflect.Type, visited map[reflect.Type]bool) map[string]any {
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
		// []byte encodes as base64 via encoding/json, not as an array of integers.
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}
		}
		return map[string]any{"type": "array", "items": reflectSchema(t.Elem(), visited)}
	case reflect.Map, reflect.Interface:
		return map[string]any{"type": "object"}
	case reflect.Struct:
		if visited[t] {
			return map[string]any{"type": "object"}
		}
		visited[t] = true
		defer delete(visited, t)
		return reflectStruct(t, visited)
	}
	return map[string]any{}
}

func reflectStruct(t reflect.Type, visited map[reflect.Type]bool) map[string]any {
	props := map[string]any{}
	var required []string
	addRequired := func(name string) {
		for _, existing := range required {
			if existing == name {
				return
			}
		}
		required = append(required, name)
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		if field.Anonymous && embeddedInlineable(field) {
			ft := field.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			embedded := reflectStruct(ft, visited)
			for k, v := range embedded["properties"].(map[string]any) {
				props[k] = v
			}
			if r, ok := embedded["required"].([]string); ok {
				for _, k := range r {
					addRequired(k)
				}
			}
			continue
		}

		if !field.IsExported() {
			continue
		}

		name := fieldName(field)
		if name == "" {
			continue
		}
		props[name] = reflectSchema(field.Type, visited)
		if isRequired(field.Tag.Get("jsonschema")) {
			addRequired(name)
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

// Limitation: a struct that anonymously embeds time.Time (e.g. `struct { time.Time }`)
// serializes via encoding/json as a single timestamp string, but this reflector will
// emit it as an object with a nested "Time" property. Use a named field instead.
func embeddedInlineable(field reflect.StructField) bool {
	tagName, _, _ := strings.Cut(field.Tag.Get("json"), ",")
	if tagName != "" {
		return false
	}
	ft := field.Type
	for ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	return ft.Kind() == reflect.Struct && ft != timeType
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
