from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (FastembedTextEmbedder,
                                                                  FastembedDocumentEmbedder,
                                                                  FastembedSparseTextEmbedder,
                                                                  FastembedSparseDocumentEmbedder)
from pypdf import PdfReader

reader = PdfReader("../data/genai/Owners_Manual_tesla.pdf")
documents = [Document(content=page.extract_text()) for page in reader.pages]

document_store = QdrantDocumentStore(
    location=":memory:",
    recreate_index=True,
    use_sparse_embeddings=True,
    embedding_dim=384
)

indexing = Pipeline()
indexing.add_component(name="sparse_doc_embedder", instance=FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1"))
indexing.add_component(name="dense_doc_embedder", instance=FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5"))
indexing.add_component(name="writer", instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
indexing.connect(sender="sparse_doc_embedder", receiver="dense_doc_embedder")
indexing.connect(sender="dense_doc_embedder", receiver="writer")

indexing.run(data={"sparse_doc_embedder": {"documents": documents}})

querying = Pipeline()
querying.add_component(name="sparse_text_embedder", instance=FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1"))
querying.add_component(name="dense_text_embedder", instance=FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5",
    prefix="Represent this sentence for searching relevant passages: "))
querying.add_component(name="retriever", instance=QdrantHybridRetriever(document_store=document_store))

querying.connect(sender="sparse_text_embedder.sparse_embedding", receiver="retriever.query_sparse_embedding")
querying.connect(sender="dense_text_embedder.embedding", receiver="retriever.query_embedding")

question = "Wie stelle ich das Lenkgewicht ein?"

results = querying.run(
    data={"dense_text_embedder": {"text": question},
          "sparse_text_embedder": {"text": question}}
)

print(results["retriever"]["documents"][0])