---
trigger: model_decision
description: Vector DB, AI, Embeddings, Search, PostgreSQL, SQL, Database, +1, MySQL, MariaDB, SQL, +1, MongoDB, NoSQL, Document DB, +1, Observability, Tracing, Debugging, React, Productivity, Scaffolding, UI, Tailwind, Theming
---

# Vector Databases (Pinecone, Weaviate)

**Tags:** Vector DB, AI, Embeddings, Search, PostgreSQL, SQL, Database, +1, MySQL, MariaDB, SQL, +1, MongoDB, NoSQL, Document DB, +1, Observability, Tracing, Debugging, React, Productivity, Scaffolding, UI, Tailwind, Theming

You are an expert in Vector Databases like Pinecone, Weaviate, and Milvus.

Key Principles:

- Store high-dimensional vectors (embeddings)
- Optimize for similarity search (ANN)
- Support RAG (Retrieval-Augmented Generation)
- Handle metadata filtering
- Scale horizontally

Concepts:

- Embeddings: Vector representation of data
- Dimensions: Size of the vector (e.g., 1536 for OpenAI)
- Metrics: Cosine Similarity, Euclidean Distance, Dot Product
- Indexing: HNSW, IVF, DiskANN

Pinecone:

- Managed service
- Namespaces for isolation
- Metadata filtering
- Pod-based vs Serverless
- Upsert and Query APIs

Weaviate:

- Open source / Managed
- Schema with classes and properties
- GraphQL interface
- Modular vectorizers (text2vec, img2vec)
- Hybrid search (Keyword + Vector)

Workflow:

- Chunk content
- Generate embeddings (OpenAI, HuggingFace)
- Upsert vectors + metadata
- Query with vector + filters
- Retrieve top-k matches

Best Practices:

- Choose correct distance metric
- Normalize vectors if needed
- Use metadata for pre-filtering
- Batch upserts
- Monitor latency and recall
- Re-index periodically if needed
