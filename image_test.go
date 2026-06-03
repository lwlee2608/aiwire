package aiwire

import (
	"encoding/base64"
	"testing"
)

func TestGeneratedImageDecode(t *testing.T) {
	raw := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)

	mime, data, err := GeneratedImage{URL: dataURL}.Decode()
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if mime != "image/png" {
		t.Errorf("mime = %q, want image/png", mime)
	}
	if string(data) != string(raw) {
		t.Errorf("data = %v, want %v", data, raw)
	}
}

func TestGeneratedImageDecodePlain(t *testing.T) {
	mime, data, err := GeneratedImage{URL: "data:text/plain,hello%20world+a"}.Decode()
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if mime != "text/plain" {
		t.Errorf("mime = %q, want text/plain", mime)
	}
	if string(data) != "hello world+a" {
		t.Errorf("data = %q, want %q", data, "hello world+a")
	}
}

func TestGeneratedImageDecodeErrors(t *testing.T) {
	cases := []string{
		"https://example.com/a.png", // not a data URL
		"data:image/png;base64",     // missing comma
		"data:image/png;base64,!!!", // invalid base64
	}
	for _, url := range cases {
		if _, _, err := (GeneratedImage{URL: url}).Decode(); err == nil {
			t.Errorf("Decode(%q): expected error, got nil", url)
		}
	}
}

func TestImageInputFromBytes(t *testing.T) {
	got := ImageInputFromBytes("image/jpeg", []byte("abc"))
	want := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString([]byte("abc"))
	if got.URL != want {
		t.Errorf("URL = %q, want %q", got.URL, want)
	}
}

func TestImageConfigMap(t *testing.T) {
	if m := imageConfigMap(ImageOption{}); m != nil {
		t.Errorf("empty option: got %v, want nil", m)
	}

	m := imageConfigMap(ImageOption{
		AspectRatio: "16:9",
		ConfigExtra: map[string]any{"seed": 7},
	})
	if m["aspect_ratio"] != "16:9" {
		t.Errorf("aspect_ratio = %v, want 16:9", m["aspect_ratio"])
	}
	if m["seed"] != 7 {
		t.Errorf("seed = %v, want 7", m["seed"])
	}
}
