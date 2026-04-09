import torch
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List

def save_colbert_index(
    embeddings_list: List[torch.Tensor],  # [(B1, N1, D), (B2, N2, D), ...]
    doc_ids: List[str],                   # Flat list [total_docs]
    output_dir: str,
    use_half: bool = True,
    index_type: str = "FlatIP",
    normalize_embeddings: bool = True
) -> None:
    """
    Save ColBERT-style embeddings to FAISS index.
    
    Args:
        embeddings_list: List of 3D tensors [(B1, N1, D), (B2, N2, D), ...]
                         Each batch is already padded to its max sequence length.
        doc_ids: Flat list of all document IDs (sum of all batch sizes)
        output_dir: Directory to save index files
        use_half: If True, convert to float16 to save 50% memory
        index_type: "FlatIP" for dot product, "FlatL2" for Euclidean
        normalize_embeddings: If True, L2-normalize before indexing
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # === Step 1: Reshape each batch from (B, N, D) → (B*N, D) ===
    embeddings_np = []
    doc_lengths = []  # Tokens per document (N for each batch)
    
    for batch_emb in embeddings_list:
        B, N, D = batch_emb.shape
        
        # Reshape: (B, N, D) → (B*N, D)
        batch_flat = batch_emb.cpu().float().numpy().reshape(-1, D)
        
        if normalize_embeddings and index_type == "FlatIP":
            faiss.normalize_L2(batch_flat)
        if use_half:
            batch_flat = batch_flat.astype(np.float16)
        
        embeddings_np.append(batch_flat)
        
        # Track: each document in this batch has N tokens
        doc_lengths.extend([N] * B)
    
    # === Step 2: Validate ===
    total_docs = sum(emb.shape[0] for emb in embeddings_list)
    assert len(doc_ids) == total_docs, \
        f"doc_ids length ({len(doc_ids)}) must match total documents ({total_docs})"
    
    D = embeddings_list[0].shape[-1]
    assert all(emb.shape[-1] == D for emb in embeddings_list), "Embedding dimensions must be uniform"
    
    # === Step 3: Build offset map (per document) ===
    doc_offsets = np.cumsum([0] + doc_lengths, dtype=np.int64)
    
    # === Step 4: Flatten all tokens into single 2D matrix ===
    all_vectors = np.vstack(embeddings_np)  # Shape: [Total_Tokens, D]
    
    # === Step 5: Build FAISS index ===
    index = faiss.IndexFlatIP(D) if index_type == "FlatIP" else faiss.IndexFlatL2(D)
    index.add(all_vectors)
    
    # === Step 6: Save ===
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    metadata = {
        "doc_ids": doc_ids,
        "doc_offsets": doc_offsets,
        "doc_lengths": doc_lengths,
        "embedding_dim": D,
        "dtype": str(all_vectors.dtype),
        "index_type": index_type,
        "num_docs": len(doc_ids),
        "num_tokens": len(all_vectors)
    }
    
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Index saved: {len(doc_ids):,} docs, {len(all_vectors):,} tokens, {D} dim")
    
# Your data structure:
embeddings_list = [
    torch.randn(32, 780, 1024),   # Batch 1: 32 docs × 780 tokens
    torch.randn(32, 772, 1024),   # Batch 2: 32 docs × 772 tokens
    torch.randn(32, 755, 1024),   # Batch 3: 32 docs × 755 tokens
]

# Flat list of all document IDs (32 + 32 + 32 = 96 total)
doc_ids = [f"doc_{i}" for i in range(96)]

save_colbert_index(
    embeddings_list=embeddings_list,
    doc_ids=doc_ids,
    output_dir="./visual_colbert_index",
    use_half=True,
    index_type="FlatIP",
    normalize_embeddings=True
)