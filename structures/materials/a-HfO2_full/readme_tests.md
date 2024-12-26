train_sparse_small.py --> sparse connectivity (rcut = 1.2), small embedding sizes (16)
train_sparse_big.py --> sparse connectivity (rcut = 1.2), large embedding sizes (16)
train_dense_small.py --> dense connectivity (rcut = 4.0), small embedding sizes (16)
train_dense_big.py --> dense connectivity (rcut = 4.0), large embedding sizes (128)

Notes: 
- high connectivity = more communication required
- large embedding sizes = more communication volume, but also more computation
- "larger cutoff" here is 4.0A, which is what we know should fit on 1 A100 (80 GB) with small embeddings
- both train_dense_small and train_dense_big run OOM on 1 A100 GPU
