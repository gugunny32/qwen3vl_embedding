import psycopg2
import psycopg2.extensions
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import json

from typing import Optional
from loguru import logger

from app.config import get_settings

# Register dict as JSON adapter globally
psycopg2.extensions.register_adapter(dict, Json)


class Database:
    def __init__(self):
        self.settings = get_settings()
        self.pool: Optional[ThreadedConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.settings.SUPABASE_DB_URL
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor from a connection"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., INSERT, UPDATE, DELETE)
                return None

    def execute_many(self, query: str, params_list: list):
        """Execute a query with multiple parameter sets"""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)

    def initialize_schema(self):
        """Initialize database schema from schema.sql"""
        import os

        schema_path = os.path.join(
            os.path.dirname(__file__),
            "schema.sql"
        )

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(schema_sql)

            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def close(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connections closed")


# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def close_database():
    """Close database connection"""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
