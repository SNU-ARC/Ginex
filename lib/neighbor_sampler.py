from typing import List, Optional, Tuple, NamedTuple, Callable
import os
import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor
import torch.multiprocessing as mp
from lib.cpp_extension.wrapper import sample


def sample_adj_ginex(rowptr, col, subset, num_neighbors, replace, cache_data=None, address_table=None):
    rowptr, col, n_id, indptr = sample.sample_adj_ginex(                                                   
        rowptr, col, subset, cache_data, address_table, num_neighbors, replace)                                 
    out = SparseTensor(rowptr=rowptr, row=None, col=col,                                                 
                       sparse_sizes=(subset.size(0), n_id.size(0)),                                      
                       is_sorted=True)                                                                   
    return out, n_id                                                                                     


def sample_adj_mmap(rowptr, col, subset: torch.Tensor, num_neighbors: int, replace: bool = False):
    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(                                         
    rowptr, col, subset, num_neighbors, replace)                                                         
    out = SparseTensor(rowptr=rowptr, row=None, col=col, value=None,                                     
                       sparse_sizes=(subset.size(0), n_id.size(0)),                                      
                       is_sorted=True)
    return out, n_id


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]


    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class GinexNeighborSampler(torch.utils.data.DataLoader):
    '''
    Neighbor sampler of Ginex. We modified NeighborSampler class of PyG.

    Args:
        indptr (Tensor): the indptr tensor.
        indices (Tensor): the (memory-mapped) indices tensor.
        exp_name (str): the name of the experiments used to designate the path of the
            runtime trace files.
        sb (int): the superbatch number.
        sizes ([int]): The number of neighbors to sample for each node in each layer. 
            If set to sizes[l] = -1`, all neighbors are included in layer `l`.
        node_idx (Tensor): The nodes that should be considered for creating mini-batches.
        cache_data (Tensor): the data array of the neighbor cache.
        address_table (Tensor): the address table of the neighbor cache.
        num_nodes (int): the number of nodes in the graph.
        transform (callable, optional): A function/transform that takes in a sampled 
            mini-batch and returns a transformed version. (default: None) 
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`, such as `batch_size`,
            `shuffle`, `drop_last`m `num_workers`.
    '''
    def __init__(self, indptr, indices, exp_name, sb,
                 sizes: List[int], node_idx: Tensor,
                 cache_data = None, address_table = None,
                 num_nodes: Optional[int] = None,
                 transform: Callable = None, **kwargs):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        self.indptr = indptr
        self.indices = indices
        self.exp_name = exp_name
        self.sb = sb
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.cache_data = cache_data
        self.address_table = address_table

        self.sizes = sizes
        self.transform = transform

        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        self.batch_count = torch.zeros(1, dtype=torch.int).share_memory_()
        self.lock = mp.Lock()
    
        super(GinexNeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = sample_adj_ginex(self.indptr, self.indices, n_id, size, False, self.cache_data, self.address_table)
            
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            adjs.append(Adj(adj_t, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        
        self.lock.acquire()
        batch_count = self.batch_count.item()
        n_id_filename = os.path.join('./trace', self.exp_name, 'sb_' + str(self.sb) + '_ids_' + str(self.batch_count.item()) + '.pth')
        adjs_filename = os.path.join('./trace', self.exp_name, 'sb_' + str(self.sb) + '_adjs_' + str(self.batch_count.item()) + '.pth')
        self.batch_count += 1
        self.lock.release()

        torch.save(n_id, n_id_filename)
        torch.save(adjs, adjs_filename)


    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class MMAPNeighborSampler(torch.utils.data.DataLoader):
    '''
    Sampler for the baseline. We modified NeighborSampler class of PyG.

    Args:
        indptr (Tensor): the indptr tensor.
        indices (Tensor): the (memory-mapped) indices tensor.
        sizes ([int]): The number of neighbors to sample for each node in each layer. 
            If set to sizes[l] = -1`, all neighbors are included in layer `l`.
        node_idx (Tensor): The nodes that should be considered for creating mini-batches.
        num_nodes (int): the number of nodes in the graph.
        transform (callable, optional): A function/transform that takes in a sampled 
            mini-batch and returns a transformed version. (default: None) 
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`, such as `batch_size`,
            `shuffle`, `drop_last`m `num_workers`.
    '''
    def __init__(self, indptr, indices,
                 sizes: List[int], node_idx: Tensor,
                 num_nodes: Optional[int] = None, 
                 transform: Callable = None, **kwargs):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        self.indptr = indptr
        self.indices = indices
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.transform = transform

        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(MMAPNeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = sample_adj_mmap(self.indptr, self.indices, n_id, size, False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]

            adjs.append(Adj(adj_t, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out


    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
