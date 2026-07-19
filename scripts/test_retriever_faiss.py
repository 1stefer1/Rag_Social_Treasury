from pathlib import Path

from src.utils.embedder import Embedder
from src.utils.vector_store import VectorStore

embedder = Embedder()

vs = VectorStore.load(Path("data/faiss_index"), name="kb_test")

q = embedder.embed(["что говорится про пункт 2"], input_type="query")[0]
res = vs.search(q, top_k=2)

print(res[0].id, res[0].score)
