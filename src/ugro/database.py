import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class Database:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to ~/.ugro/ugro.db
            db_dir = Path.home() / ".ugro"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir / "ugro.db"
        else:
            self.db_path = Path(db_path)
            
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                model_name TEXT,
                dataset_name TEXT,
                config TEXT,  -- JSON string
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                logs_path TEXT,
                worker_nodes TEXT  -- JSON list of nodes
            )
            """)
            
            conn.commit()
            
    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return dict-like objects
        return conn
