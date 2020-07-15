#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Apr 8, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Functions for sparse matrix manipulations in pytorch
'''
import torch
import sys
import os
import gc
import psutil


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


def memoryStats(device='cuda'):
    print('cpu %',psutil.cpu_percent())
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
    print('memory used GB: %.2f' % memoryUse)
    
    if device == 'cuda':
        print('available gpu: %.2f' % (torch.cuda.get_device_properties(device).total_memory / 2. ** 30))
        print('used gpu: %.2f' % (torch.cuda.memory_allocated() / 2. ** 30))


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            
            
def load_predicates():
    relationid2label = {}
    for p in properties['results']['bindings']:
        _id = p['property']['value'].split('/')[-1]
    #     print(_id)
        label = p['propertyLabel']['value']
        relationid2label[_id] = label

    # all unique predicate labels
    all_predicate_labels = list(relationid2label.values())
    all_predicate_ids = [kg.string_to_global_id(PREFIX_P + p, TripleComponentRole.PREDICATE) for p in list(relationid2label.keys())]
    all_predicate_labels = [p for i, p in enumerate(all_predicate_labels) if all_predicate_ids[i] != 0]
    all_predicate_ids = [p for p in all_predicate_ids if p != 0]
    assert len(all_predicate_labels) == len(all_predicate_ids)
    print(all_predicate_labels)
