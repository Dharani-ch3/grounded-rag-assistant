"""Optional production backends. Each one implements the same small interface as the
in-memory default, so the assistant code doesn't change when you switch:

* `pinecone.PineconeStore` and `weaviate.WeaviateStore` implement `retrieval.VectorStore`
  and push the access filter down into the database query.
"""
