package db

import (
	"database/sql"
	"embed"
	"log/slog"

	"github.com/jackc/pgx/v5"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

//go:embed migrations/*.sql
var embedMigrations embed.FS

func RunMigrations(dbURL, schema string) error {
	slog.Info("Running database migrations...")

	if schema == "" {
		schema = "public"
	}

	db, err := sql.Open("pgx", dbURL)
	if err != nil {
		return err
	}
	defer db.Close()

	if err := ensureSchemaExists(db, schema); err != nil {
		return err
	}

	goose.SetBaseFS(embedMigrations)

	if err := goose.SetDialect("postgres"); err != nil {
		return err
	}

	if err := goose.Up(db, "migrations"); err != nil {
		return err
	}

	slog.Info("Database migrations completed successfully")
	return nil
}

func ensureSchemaExists(db *sql.DB, schema string) error {
	query := "CREATE SCHEMA IF NOT EXISTS " + pgx.Identifier{schema}.Sanitize()
	if _, err := db.Exec(query); err != nil {
		return err
	}
	slog.Info("Schema is ready", "schema", schema)

	setPathQuery := "SET search_path TO " + pgx.Identifier{schema}.Sanitize()
	if _, err := db.Exec(setPathQuery); err != nil {
		return err
	}
	slog.Info("Set search_path", "schema", schema)

	return nil
}
