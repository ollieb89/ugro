from datetime import datetime
import json
from typing import Dict, List, Optional
import uuid
from .database import Database

class JobQueue:
    def __init__(self, db: Database):
        self.db = db
        
    def enqueue_job(self, model_name: str, dataset_name: str, config: Dict, priority: int = 0) -> str:
        """Add a job to the queue"""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        created_at = datetime.now().isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO jobs (id, priority, status, model_name, dataset_name, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, priority, 'pending', model_name, dataset_name, json.dumps(config), created_at)
            )
            conn.commit()
            
        return job_id
        
    def peek_next_job(self) -> Optional[Dict]:
        """Get the next pending job with highest priority"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Order by priority (descending) then creation time (ascending)
            cursor.execute(
                """
                SELECT * FROM jobs 
                WHERE status = 'pending' 
                ORDER BY priority DESC, created_at ASC 
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            
            if row:
                job = dict(row)
                job['config'] = json.loads(job['config'])
                return job
            return None
            
    def update_status(self, job_id: str, status: str, worker_nodes: List[str] = None):
        """Update job status"""
        timestamp = datetime.now().isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            params = [status]
            
            if status == 'running':
                updates.append("started_at = ?")
                params.append(timestamp)
                if worker_nodes:
                    updates.append("worker_nodes = ?")
                    params.append(json.dumps(worker_nodes))
            elif status in ['completed', 'failed']:
                updates.append("completed_at = ?")
                params.append(timestamp)
                
            params.append(job_id)
            
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

    def list_jobs(self, limit: int = 10) -> List[Dict]:
        """List most recent jobs"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM jobs 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get specific job details"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job = dict(row)
                job['config'] = json.loads(job['config'])
                if job['worker_nodes']:
                    job['worker_nodes'] = json.loads(job['worker_nodes'])
                return job
            return None
