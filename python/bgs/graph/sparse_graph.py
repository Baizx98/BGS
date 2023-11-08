"""Compressed sparse matrix representation of graphs"""
import torch as th
from torch_sparse import coalesce

import warnings

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class CSRGraph:
    def __init__(self, edge_index=None) -> None:
        self._indptr: th.Tensor = None
        self._indices: th.Tensor = None
        if edge_index is not None:
            max_src_id = th.max(edge_index[0])
            max_dst_id = th.max(edge_index[1])
            max_id = max(max_src_id, max_dst_id)
            adj_coo = th.sparse_coo_tensor(
                edge_index,
                th.ones(edge_index.shape[1]),
                size=[max_id + 1, max_id + 1],
            )
            adj_csr = adj_coo.to_sparse_csr()
            self._indptr = adj_csr.crow_indices()
            self._indices = adj_csr.col_indices()

    @property
    def indptr(self):
        return self._indptr

    @property
    def indices(self):
        return self._indices

    @property
    def out_degrees(self):
        return self._indptr[1:] - self._indptr[:-1]

    @property
    def node_count(self):
        return self._indptr.shape[0] - 1

    @property
    def edge_count(self):
        return self._indices.shape[0]

    @property
    def out_neighbors(self, nid: th.Tensor) -> th.Tensor:
        return self.indices[self._indptr[nid] : self.indptr[nid + 1]]

    def share_memory(self):
        self._indptr.share_memory_()
        self._indices.share_memory_()


class COOGraph:
    def __init__(self, edge_index=None) -> None:
        self._indices: th.Tensor = None
        if edge_index is not None:
            adj_coo = th.sparse_coo_tensor(edge_index, th.ones(edge_index).shape[1])
            self._indices = adj_coo.indices
            self._node_count: int = adj_coo.size[1]

    @property
    def indices(self):
        return self._indices

    @property
    def node_count(self):
        return self._node_count

    def share_memory(self):
        self._indices.share_memory_()
