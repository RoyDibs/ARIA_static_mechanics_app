import os
import json
import pickle
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(f"sentence-transformers not available: {e}")


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("create_e5_embeddings_pickle")


def read_all_txt(root_dir: str) -> Dict[str, str]:
    texts = {}
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.txt'):
                fp = os.path.join(r, f)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        texts[fp] = fh.read()
                except Exception as e:
                    log.warning(f"Failed reading {fp}: {e}")
    if not texts:
        raise RuntimeError(f"No .txt files found under {root_dir}")
    return texts


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def derive_metadata(fp: str) -> Dict[str, str]:
    p = Path(fp)
    parts = p.parts
    content_type = "unknown"
    topic = p.stem
    # infer content type from path components
    if "course_slides" in parts:
        content_type = "course_slide"
    elif "exercises" in parts:
        content_type = "exercise_question"
    return {
        "source_file": p.name,
        "content_type": content_type,
        "topic": topic
    }


def build_embeddings_pickle(processed_root: str, output_pickle: str, model_name: str = "intfloat/e5-large-v2"):
    log.info(f"Loading processed content from: {processed_root}")
    texts = read_all_txt(processed_root)

    log.info(f"Preparing chunks...")
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    for fp, content in texts.items():
        base = Path(fp).stem
        meta_base = derive_metadata(fp)
        for i, ch in enumerate(chunk_text(content)):
            ids.append(f"{base}_chunk_{i}")
            documents.append(ch)
            metadatas.append(meta_base)

    log.info(f"Total chunks: {len(documents)} from {len(texts)} files")

    log.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    log.info("Encoding chunks with e5-large-v2...")
    # batch encode for efficiency
    embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=False)
    embeddings = embeddings.astype(np.float32).tolist()

    data = {
        'documents': documents,
        'embeddings': embeddings,
        'metadatas': metadatas,
        'ids': ids
    }

    outp = Path(output_pickle)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as fh:
        pickle.dump(data, fh)

    size_mb = outp.stat().st_size / (1024 * 1024)
    log.info(f"Saved pickle: {outp} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create embeddings_data.pkl using e5-large-v2")
    parser.add_argument('--processed_root', required=True, help='Root of processed_content (contains course_slides/, exercises/, etc.)')
    parser.add_argument('--output_pickle', required=True, help='Path to write embeddings_data.pkl')
    parser.add_argument('--model', default='intfloat/e5-large-v2')
    args = parser.parse_args()

    build_embeddings_pickle(args.processed_root, args.output_pickle, args.model)

