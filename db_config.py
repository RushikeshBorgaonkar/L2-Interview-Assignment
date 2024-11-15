import os
from dotenv import load_dotenv
import psycopg2
import numpy as np

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432')
        )
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_id VARCHAR(255),
                    text_content TEXT,
                    embedding vector(384)
                );
            """)
            self.conn.commit()

    def add_embedding_to_db(self, embedding, text_id, text_content):
        with self.conn.cursor() as cur:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            cur.execute(
                "INSERT INTO embeddings (text_id, text_content, embedding) VALUES (%s, %s, %s)",
                (text_id, text_content, embedding_list)
            )
            self.conn.commit()

    def search_similar_vectors(self, query_embedding, top_k=2):
        with self.conn.cursor() as cur:
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            cur.execute("""
                SELECT text_content, 1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_list, query_list, top_k))
            
            results = cur.fetchall()
            return [(text, float(score)) for text, score in results]

    def close(self):
        self.conn.close()

    def clear_embeddings(self):
        """Clear all existing embeddings from the database"""
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE embeddings")
        self.conn.commit()
