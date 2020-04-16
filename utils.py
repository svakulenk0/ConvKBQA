#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Apr 8, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Functions for sparse matrix manipulations in pytorch
'''

import torch


# adopted from https://github.com/pbloem/gated-rgcn/blob/a3dfa1cb162e2050c31f6e54bc21f0b6363bded2/kgmodels/util/util.py#L59
def adj(adjacencies, num_nodes, num_relations, vertical=False, include_inverse=True):
    """
    Creates a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically by default) for a CPU.
    :param adjacencies: list of lists of tuples representing the edges
    :param num_nodes: maximum number of nodes in the graph
    :return: sparse tensor
    """
    ST = torch.sparse.FloatTensor

    r, n = num_relations, num_nodes
    size = (r*n, n) if vertical else (n, r*n)

    from_indices = []
    upto_indices = []
    
    relation_mask = []  # vector indicating predicates in adjacency matrix
    for rel, edges in enumerate(adjacencies):
        offset = rel * n
        
        # duplicate edges in the opposite direction
        if include_inverse:
            edges.extend([x[::-1] for x in edges])

        relation_mask.extend([rel] * len(edges))
        for fr, to in edges:

            if vertical:
                fr = offset + fr
            else:
                to = offset + to

            from_indices.append(fr)
            upto_indices.append(to)
    
    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long)
    
    relation_mask = torch.tensor(relation_mask)

    return indices, relation_mask