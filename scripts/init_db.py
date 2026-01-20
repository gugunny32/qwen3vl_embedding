#!/usr/bin/env python3
"""
Database initialization script.
Initializes the database schema for the multimodal RAG system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import get_database
from loguru import logger


def main():
    """Initialize database schema"""
    logger.info("Initializing database schema...")

    try:
        db = get_database()
        db.initialize_schema()
        logger.info("Database schema initialized successfully!")

        # Verify tables exist
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()

            logger.info("Tables in database:")
            for table in tables:
                logger.info(f"  - {table['table_name']}")

        # Check extensions
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT extname
                FROM pg_extension
                WHERE extname IN ('vector', 'pg_trgm', 'uuid-ossp');
            """)
            extensions = cursor.fetchall()

            logger.info("Installed extensions:")
            for ext in extensions:
                logger.info(f"  - {ext['extname']}")

        logger.info("Database initialization complete!")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
