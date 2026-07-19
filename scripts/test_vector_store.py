from pathlib import Path

from src.utils.embedder import Embedder
from src.utils.vector_store import StoredChunk, VectorStore

embedder = Embedder()
texts = ["пункт 1. что-то", "пункт 2. другое"]
vecs = embedder.embed(texts, input_type="document")

vs = VectorStore(dim=vecs.shape[1])
vs.add(vecs, [StoredChunk(id=f"c{i}", text=t, meta={}) for i, t in enumerate(texts)])

q = embedder.embed(["что говорится про пункт 2"], input_type="query")[0]
res = vs.search(q, top_k=2)
print(res[0].id, res[0].score)

vs.save(Path("data/faiss_index"), name="kb_test")
