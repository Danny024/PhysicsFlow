-- PhysicsFlow v2.0 — PostgreSQL initialisation script
-- Runs automatically when the postgres container is first started.
-- SQLAlchemy / Alembic will create the actual application tables on
-- first engine boot; this script just sets up the database and role.

-- Enable the uuid-ossp extension so gen_random_uuid() is available.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram index for well-name search

-- Grant full privileges to the application user (already created by
-- the POSTGRES_USER env var, but the GRANT is idempotent).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = current_user
    ) THEN
        -- Should not happen; container env creates the user.
        RAISE NOTICE 'Role % already exists', current_user;
    END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE physicsflow TO physicsflow;
