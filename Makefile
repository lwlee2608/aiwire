GO ?= $(shell which go 2>/dev/null)

.PHONY: all help build clean test test-race vet fmt \
	test-openai test-openrouter test-zai test-cerebras test-integration test-usage test-reasoning

all: clean build test

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  all              Clean, build, and run unit tests"
	@echo "  build            Compile the library (go build ./...)"
	@echo "  vet              Run go vet"
	@echo "  fmt              Run gofmt -w on all sources"
	@echo "  test             Run unit tests (no network)"
	@echo "  test-race        Run unit tests with the race detector"
	@echo "  test-openai      Run integration tests against OpenAI       (needs OPENAI_API_KEY)"
	@echo "  test-openrouter  Run integration tests against OpenRouter   (needs OPENROUTER_API_KEY)"
	@echo "  test-zai         Run integration tests against Z.ai         (needs ZAI_API_KEY)"
	@echo "  test-cerebras    Run integration tests against Cerebras    (needs CEREBRAS_API_KEY)"
	@echo "  test-usage       Run usage/cache-token tests                (needs OPENAI_API_KEY and OPENROUTER_API_KEY)"
	@echo "  test-reasoning   Run reasoning integration tests           (needs OPENROUTER_API_KEY)"
	@echo "  test-integration Run all integration tests"
	@echo "  clean            Clear the Go test cache"

build:
	$(GO) build ./...

vet:
	$(GO) vet ./...

fmt:
	$(GO) fmt ./...

clean:
	$(GO) clean -testcache

test:
	$(GO) test -v ./...

test-race:
	$(GO) test -v -race ./...

test-openai:
	$(GO) test -v -tags=integration -count=1 -run '^TestOpenAI_' ./integration/...

test-openrouter:
	$(GO) test -v -tags=integration -count=1 -run '^(TestOpenRouter|TestAgent)_' ./integration/...

test-zai:
	$(GO) test -v -tags=integration -count=1 -run '^TestZAI_' ./integration/...

test-cerebras:
	$(GO) test -v -tags=integration -count=1 -run '^TestCerebras_' ./integration/...

test-usage:
	$(GO) test -v -tags=integration -count=1 -run '^TestUsage_' ./integration/...

test-reasoning:
	$(GO) test -v -tags=integration -count=1 -run '^TestReasoning_' ./integration/...

test-integration: test-openai test-openrouter test-zai test-cerebras test-reasoning
