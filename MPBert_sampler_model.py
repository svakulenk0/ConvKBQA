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

from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from hdt import HDTDocument

from settings import *
from utils import *


def build_look_up(keys):
    look_up_table = {}
    for i, k in enumerate(keys):
        look_up_table[k.item()] = i
    return look_up_table


def look_up(look_up_table, subset, scores):
    values = []
    default_value_idx = len(scores)
    for i, key in enumerate(subset):
        if key in look_up_table:
            value = look_up_table[key]
        else:
            # fill missing values with default
            value = default_value_idx
        values.append(value)
    assert len(values) == len(subset)
    positions = torch.tensor(values).unsqueeze(-1)
    
    # add 0 for entities that were not scored
    s_scores = torch.zeros(size=(scores.shape[0]+1, 1))
    s_scores[:scores.shape[0]] = scores

    return s_scores.gather(0, positions)


class SamplingLayer(nn.Module):
    def __init__(self, kg, topk_entities):
        super(SamplingLayer, self).__init__()
        self.kg = kg
        self.ksample = topk_entities
        
    def forward(self, e_scores, entity_ids, p_scores, predicate_ids):
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
        sampled_entities = entity_ids[indices[:self.ksample]].tolist()  # choose top-k matching entities
#         print("Retrieving adjacencies for %d entities"%len(sampled_entities))

        # build a lookup table for entity & predicate scores
        e_table = build_look_up(entity_ids)
        p_table = build_look_up(predicate_ids)

        # sample predicates?
        sampled_predicates = []  # predicate_ids.tolist()
        
        # request kg through hdt api for a subgraph given entity and relation subsets
        self.kg.configure_hops(1, sampled_predicates, 'predef-wikidata2018-09-all', True, False)
        s_entity_ids, s_predicate_ids, adjacencies = self.kg.compute_hops(sampled_entities, 50000, 0)
        
        print("Retrieved new subgraph with %d entities and %d relations"%(len(s_entity_ids), len(s_predicate_ids)))

        # load subgraph into tensor
        indices, relation_mask = adj(adjacencies, len(s_entity_ids), len(s_predicate_ids))

        # lookup local scores to activate respective entities & predicates
        e_scores = look_up(e_table, s_entity_ids, e_scores)
        p_scores = look_up(p_table, s_predicate_ids, p_scores)
        
        del adjacencies, e_table, p_table, s_predicate_ids
        
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
        
        # propagate score from the Transformer output over predicate labels to predicate ids
        p_scores = p_scores.gather(0, relation_mask)
        subgraph = torch.sparse.FloatTensor(indices=indices, values=p_scores,
                                            size=(num_entities, num_entities*num_relations))
        
        
        # MP step: propagates entity activations to the adjacent nodes
        y = torch.sparse.mm(subgraph.t(), e_scores)
        
        del subgraph, indices, e_scores, p_scores, 

        # and MP to entities summing over all relations
        y = y.view(num_relations, num_entities).sum(dim=0) # new dim for the relations
        
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

    def __init__(self, config, hdt_file='wikidata2018_09_11.hdt', topk_entities=10, bottleneck_dim=32):
        super(MessagePassingHDTBert, self).__init__(config)
        
        # entity matching Transformer
        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.pre_classifier = nn.Linear(config.hidden_size, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, self.config.num_labels)
        
        # initialise connection to the Wikidata KG through the HDT API
        kg = HDTDocument(hdt_path+hdt_file)
        # sampling layer with subgraph retrieval
        self.subgraph_sampling = SamplingLayer(kg, topk_entities)
        
        # predicted scores are propagated via MP layer into the entity subser distribution defined by the subgraph
        self.mp = MPLayer()
        
        self.init_weights()

    def forward(
        self,
        entities_q=None,
        predicates_q=None,
        answer=None,
    ):
        with torch.no_grad():
            # predict entity matches
            input_ids, token_type_ids, attention_mask, entity_ids = entities_q
            print("Ranking %d entities"%(len(input_ids)))
            e_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask
            )
            hidden_state = e_outputs[0]
            e_outputs = hidden_state[:, 0]  # pooled output
            
            # predict predicate matches
            input_ids2, token_type_ids2, attention_mask2, predicate_ids = predicates_q
            print("Ranking %d predicates"%(len(input_ids2)))
            p_outputs = self.bert(
                input_ids2,
                attention_mask=attention_mask2
            )
            hidden_state = p_outputs[0]
            p_outputs = hidden_state[:, 0]  # pooled output
            
        e_outputs = self.dropout(e_outputs)
        # introduced bottleneck before the classifier
        e_outputs = self.pre_classifier(e_outputs)
        entity_logits = self.classifier(e_outputs)
        del input_ids, token_type_ids, attention_mask, e_outputs
        
        
        p_outputs = self.dropout(p_outputs)
        # introduced bottleneck before the classifier
        p_outputs = self.pre_classifier(p_outputs)
        predicate_logits = self.classifier(p_outputs)
        del input_ids2, token_type_ids2, attention_mask2, p_outputs
        
        
        # sampling layer: pick top-scored entities and relations, retrieve the subgraph and load into tensor
        print("Start sampling")
        subgraph, entity_ids = self.subgraph_sampling(entity_logits, entity_ids,
                                                      predicate_logits, predicate_ids)
        
        # MP layer takes predicate scores and propagates them to the adjacent entities
        logits = self.mp(subgraph)
        del subgraph
        
        # candidate answers are all entities in the subgraph
        num_entities = len(entity_ids)
        
        outputs = (logits,)

        if answer is not None:
            # terminate with 0-loss if correct answer is not in the subgraph
            answer_id = answer.item()
            # print(answer_id)
            if answer_id not in entity_ids:
                print("Wrong subgraph selected")
                loss = torch.tensor(0)
            else:
                print("Correct subgraph selected")
                answer_idx = torch.tensor([entity_ids.index(answer_id)])
                del entity_ids
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_entities), answer_idx)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits