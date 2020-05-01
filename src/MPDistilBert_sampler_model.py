#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Apr 4, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Transformer for sequence classification with a message-passing layer 
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from hdt import HDTDocument

from settings import *
from utils import *
from all_predicate_ids import all_predicate_ids


def build_look_up(keys):
    look_up_table = {}
    for i, k in enumerate(keys):
#         look_up_table[k.item()] = i
        look_up_table[k] = i
    return look_up_table


def look_up(look_up_table, subset, scores, top_p=None):
    values = []
    default_value_idx = len(scores)
#     print(subset)
    for i, key in enumerate(subset):
        if key in look_up_table:
            value = look_up_table[key]
        else:
            # fill missing values with default
            value = default_value_idx
        values.append(value)
    assert len(values) == len(subset)
    positions = torch.tensor(values).unsqueeze(-1)
    del values
    
    # add 0 for entities that were not scored
    s_scores = torch.zeros(size=(scores.shape[0]+1, 1))
    if top_p:
        # sparsify scores by keeping top p predicates
        weights, indices = torch.sort(scores.view(-1), descending=True)
        s_scores.scatter_(0, indices[:top_p], weights[:top_p])
    else:
        # keep all scores
        s_scores[:scores.shape[0]] = scores
    return s_scores.gather(0, positions)


class SamplingLayer(nn.Module):
    def __init__(self, hdt_path, topk_entities, topk_predicates):
        super(SamplingLayer, self).__init__()
        self.hdt_path = hdt_path
        self.top_e = topk_entities
        self.top_p = topk_predicates
        
    def forward(self, e_scores, entity_ids, p_scores, answer=None, all_predicate_ids=all_predicate_ids):
        '''
        Inputs:
            *e_scores*: entity scores from Transformer
            *entity_ids*: global entity ids to request the KG for adjacencies
            *p_scores*: predicate scores from Transformer
        Outputs:
            *subgraph*: subgraph edges and entities
        '''
#         with torch.autograd.detect_anomaly():
            # get the top-k (predicates/)entities based on the score vectors
        weights, indices = torch.sort(e_scores.view(-1), descending=True)
        sampled_entities = entity_ids[indices[:self.top_e]].tolist()  # choose top-k matching entities
#         print("Retrieving adjacencies for %d entities"%len(sampled_entities))
        # sample predicates?
        sampled_predicates = []  # predicate_ids.tolist()
#         weights, indices = torch.sort(p_scores.view(-1), descending=True)
#         sampled_predicates = predicate_ids[indices[:self.top_p]].tolist()
        
        with torch.no_grad():
        
            # initialise connection to the Wikidata KG through the HDT API
            kg = HDTDocument(self.hdt_path)
            # request kg through hdt api for a subgraph given entity and relation subsets
            kg.configure_hops(1, sampled_predicates, 'predef-wikidata2018-09-all', True, False)
            s_entity_ids, s_predicate_ids, adjacencies = kg.compute_hops(sampled_entities, 5000, 0)
            kg.remove()
            del kg
    #         print("Retrieved new subgraph with %d entities and %d relations" % (len(s_entity_ids), len(s_predicate_ids)))
            
            # check subgraph exists
            if not s_entity_ids:
                return (), None
            
            # check we are in the right subgraph
            if answer is not None and answer not in s_entity_ids:
                return (), None

            # build a lookup table for entity & predicate scores
            e_table = build_look_up(entity_ids)
            p_table = build_look_up(all_predicate_ids)
            del all_predicate_ids
        
        # load subgraph into tensor
        indices, relation_mask = adj(adjacencies, len(s_entity_ids), len(s_predicate_ids))
#         print("%d triples" % len(indices))
        
        # lookup local scores to activate respective entities & predicates
        e_scores = look_up(e_table, s_entity_ids, e_scores)
        p_scores = look_up(p_table, s_predicate_ids, p_scores)
        del p_table, s_predicate_ids, e_table, adjacencies
        
        # clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        return (indices, e_scores, p_scores, relation_mask), s_entity_ids


class MPLayer(nn.Module):
    def __init__(self):
        super(MPLayer, self).__init__()

    def forward(self, subgraph):
        '''
        Inputs:
            *p_scores*: predicate scores from Transformer
        Outputs:
            *y*: answer activations
        '''
        
        indices, e_scores, p_scores, relation_mask = subgraph
        
        num_entities = e_scores.shape[0]
        num_relations = p_scores.shape[0]
#         relation_mask = relation_mask.unsqueeze(-1)
        p_scores = p_scores.view(-1)
#         print(p_scores)
                          
        # propagate score from the Transformer output over predicate labels to predicate ids
        p_scores = p_scores.gather(0, relation_mask)
        subgraph = torch.sparse.FloatTensor(indices=indices, values=p_scores,
                                            size=(num_entities, num_entities*num_relations))
        
        # MP step: propagates entity activations to the adjacent nodes
        y = torch.sparse.mm(subgraph.t(), e_scores)
        
        # and MP to entities summing over all relations
        y = y.view(num_relations, num_entities).sum(dim=0) # new dim for the relations
        del p_scores, subgraph, e_scores, indices, relation_mask
        
        return y


class MessagePassingHDTBert(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = ...
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, hdt_file='wikidata2018_09_11.hdt', topk_entities=20, topk_predicates=50, bottleneck_dim=32, seq_classif_dropout=0.9):
        super(MessagePassingHDTBert, self).__init__(config)
        
        # entity matching Transformer
        self.bert = DistilBertModel(config)
#         self.pre_classifier = nn.Linear(config.hidden_size, bottleneck_dim)

        # initialise weights for the linear layer to select a few
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(seq_classif_dropout)
        
        # sampling layer with subgraph retrieval
        self.subgraph_sampling = SamplingLayer(hdt_path+hdt_file, topk_entities, topk_predicates)
        
        # predicted scores are propagated via MP layer into the entity subser distribution defined by the subgraph
        self.mp = MPLayer()
        
        self.init_weights()

    def forward(
        self,
        entities_q=None,
        predicates_q=None,
        answer=None,
        first_question=None
    ):
            # predict entity matches
#         input_ids, token_type_ids, attention_mask, entity_ids = entities_q
#         print("Ranking %d entities"%(len(entities_q[0])))
        e_outputs = self.bert(
            entities_q[0],
            attention_mask=entities_q[2]
        )
        hidden_state = e_outputs[0]
        pooled_output = hidden_state[:, 0]
        e_outputs = self.dropout(pooled_output)
        entity_logits = self.classifier(e_outputs)
        
#         print("Ranking %d predicates"%(len(predicates_q[0])))
        p_outputs = self.bert(
            predicates_q[0],
            attention_mask=predicates_q[2]
        )
        hidden_state = p_outputs[0]
        pooled_output = hidden_state[:, 0]
        p_outputs = self.dropout(pooled_output)
        predicate_logits = self.classifier(p_outputs)

        
        if answer is not None:
            answer = answer.item()

        # sampling layer: pick top-scored entities and relations, retrieve the subgraph and load into tensor
        subgraph, entity_ids = self.subgraph_sampling(entity_logits, entities_q[-1],
                                                      predicate_logits, answer)
        
        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        # terminate with 0-loss if correct answer is not in the subgraph
        if entity_ids is None:
            loss = None
            return (None, None, None)

        # MP layer takes predicate scores and propagates them to the adjacent entities
        logits = self.mp(subgraph)
        del subgraph
        
        # candidate answers are all entities in the subgraph
        num_entities = len(entity_ids)

        outputs = logits, entity_ids

        if answer is not None:
            answer_idx = torch.tensor([entity_ids.index(answer)])
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_entities), answer_idx.view(-1))
            outputs = (loss,) + outputs
        
        del logits, entity_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs  # (loss), logits