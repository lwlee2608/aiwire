GO = $(shell which go 2>/dev/null)

.PHONY: test install generate

test:
	$(GO) test -v ./...

install:
	$(GO) install github.com/sqlc-dev/sqlc/cmd/sqlc@v1.29.0

generate: install
	sqlc generate
