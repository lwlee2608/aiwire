package aiwire

import (
	"reflect"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/shared"
)

var timeType = reflect.TypeOf(time.Time{})

func generateSchema[T any]() map[string]any {
	var v T
	return reflectSchema(reflect.TypeOf(v), map[reflect.Type]bool{})
}

func GenerateSchema[T any]() any {
	return generateSchema[T]()
}

func GenerateFunctionParameters[T any]() shared.FunctionParameters {
	return generateSchema[T]()
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
		propSchema := reflectSchema(field.Type, visited)
		isRequired, enum := parseJSONSchemaTag(field.Tag.Get("jsonschema"))
		if len(enum) > 0 {
			propSchema["enum"] = enum
		}
		if desc := field.Tag.Get("jsonschema_description"); desc != "" {
			propSchema["description"] = desc
		}
		props[name] = propSchema
		if isRequired {
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

// parseJSONSchemaTag parses the `jsonschema:"..."` struct tag. Recognized entries:
//   - "required"   — marks the field as required
//   - "enum=<val>" — repeat per allowed value (e.g. `jsonschema:"enum=a,enum=b"`)
//
// Enum values are always returned as strings and emitted as-is into the schema,
// so they are only well-formed for string-typed fields. Numeric or boolean enums
// would need type coercion and are not supported.
func parseJSONSchemaTag(tag string) (required bool, enum []string) {
	if tag == "" {
		return
	}
	for _, part := range strings.Split(tag, ",") {
		p := strings.TrimSpace(part)
		if p == "required" {
			required = true
			continue
		}
		if v, ok := strings.CutPrefix(p, "enum="); ok {
			enum = append(enum, v)
		}
	}
	return
}
