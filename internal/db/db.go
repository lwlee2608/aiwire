package db

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Config struct {
	URL      string `mask:"true"`
	Schema   string
	MaxConns int32
	MinConns int32
}

func InitDB(ctx context.Context, cfg Config) (*pgxpool.Pool, error) {
	poolConfig, err := pgxpool.ParseConfig(cfg.URL)
	if err != nil {
		return nil, fmt.Errorf("unable to parse database config: %w", err)
	}

	if cfg.MaxConns > 0 {
		poolConfig.MaxConns = cfg.MaxConns
	}

	if cfg.MinConns > 0 {
		poolConfig.MinConns = cfg.MinConns
	}

	schema := cfg.Schema
	if schema != "" {
		sanitizedSchema := pgx.Identifier{schema}.Sanitize()

		poolConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
			_, err := conn.Exec(ctx, fmt.Sprintf("SET search_path TO %s", sanitizedSchema))
			if err != nil {
				return err
			}
			return nil
		}

		slog.Info("Setting search_path for connection pool", "schema", schema)
	}

	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("unable to ping database: %w", err)
	}

	slog.Info("Connected to PostgreSQL")

	return pool, nil
}
