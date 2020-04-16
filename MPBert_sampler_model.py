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

from transformers.modeling_bert import BertPreTrainedModel, BertModel
from hdt import HDTDocument

from settings import *
from utils import adj


def build_look_up(keys, values):
    look_up_table = {}
    for i, k in enumerate(keys):
        look_up_table[k.item()] = values[i].item()
    return look_up_table


def look_up(look_up_table, subset, size=None, default_value=0):
    if not size:
        size = len(subset)
    # pads to size with 0s
    s_values = torch.zeros(size, 1)
    for i, key in enumerate(subset):
        if key in look_up_table:
            value = look_up_table[key]
        else:
            value = default_value
        s_values[[i]] = value
    return s_values


class SamplingLayer(nn.Module):
    def __init__(self, kg, topk_entities=5):
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
        
        # get the top-k (predicates/)entities based on the score vectors
        weights, indices = torch.sort(e_scores, descending=True)
        top_scores = weights[:self.ksample]
        top_entities = entity_ids[indices[:self.ksample]]  # choose top-k matching entities
        sampled_entities = [e.item() for e in top_entities]
        
        
        # build a lookup table for entity scores
        e_table = build_look_up(top_entities, top_scores)
        sampled_entities = list(e_table.keys())
        
        # build a lookup table for predicate scores
        p_table = build_look_up(predicate_ids, p_scores)

        # ? map the selected subset from the score vector indices to predicate/entity global ids
        # sampled_predicates
        
        # request kg through hdt api for a subgraph given entity and relation subsets
        sampled_predicates = []  # consider all predicates without subsampling -> bigger subgraph retrieved
        self.kg.configure_hops(1, sampled_predicates, 'predef-wikidata2018-09-all', True, False)
        s_entity_ids, s_predicate_ids, adjacencies = self.kg.compute_hops(sampled_entities, 50000000, 0)
        
        # load subgraph into tensor
        indices, relation_mask = adj(adjacencies, len(s_entity_ids), len(s_predicate_ids))
        
        # lookup local entity scores to activate entities
        e_scores = look_up(e_table, s_entity_ids)
        
        # lookup local predicate scores to activate predicates
        p_scores = look_up(p_table, s_predicate_ids)
        
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
#         e_scores = e_scores.view(-1)
        p_scores = p_scores.view(-1)
        
        # propagate score from the Transformer output over predicate labels to predicate ids
        p_scores = p_scores.gather(0, relation_mask)
        subgraph = torch.sparse.FloatTensor(indices=indices, values=p_scores,
                                            size=(num_entities, num_entities*num_relations))
        
        
        # MP step: propagates entity activations to the adjacent nodes
        y = torch.sparse.mm(subgraph.t(), e_scores)

        # and MP to entities summing over all relations
        y = y.view(num_relations, num_entities).sum(dim=0) # new dim for the relations
        
        return y


class MessagePassingHDTBert(BertPreTrainedModel):
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

    def __init__(self, config, num_entities, num_relations, weight=None, mp_layer=True, hdt_file='wikidata2018_09_11.hdt'):
        super(MessagePassingHDTBert, self).__init__(config)
        self.mp_layer = mp_layer
        if self.mp_layer:
            # the output layer is the distribution over answer-entities
            self.num_labels = num_entities
        else:
            # the output layer is the distribution over relation labels
            self.num_labels = config.num_labels
        
        # entity matching Transformer
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        # initialise connection to the Wikidata KG through the HDT API
        
        # sampling layer with subgraph retrieval
        kg = HDTDocument(hdt_path+hdt_file)
        self.subgraph_sampling = SamplingLayer(kg)
        
        # the predicted scores are then propagated via a message-passing layer into the entity subser distribution defined by the subgraph
        self.mp = MPLayer()
        
        # the output is a vector with the score distribution over the input entities
        self.weight = weight

        self.init_weights()

    def forward(
        self,
        entities_q=None,
        predicates_q=None,
        answer=None,
    ):
        
        # predict entity matches
        input_ids, token_type_ids, attention_mask, entity_ids = entities_q
        e_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,  # token types or position ids for separating concatenated input strings?? TODO
            head_mask=None,  # what's that? TODO
        )
        e_outputs = self.dropout(e_outputs[1])
        entity_logits = self.classifier(e_outputs)
        
        # predict predicate matches
        input_ids2, token_type_ids2, attention_mask2, predicate_ids = predicates_q
        p_outputs = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=None,
            head_mask=None,
        )
        p_outputs = self.dropout(p_outputs[1])
        predicate_logits = self.classifier(p_outputs)
        
        # sampling layer: pick top-scored entities and relations, retrieve the subgraph and load into tensor
        subgraph, entity_ids = self.subgraph_sampling(entity_logits.view(-1), entity_ids,
                                                      predicate_logits.view(-1), predicate_ids)
        
        num_labels = len(entity_ids)
        answer_idx = torch.tensor([entity_ids.index(answer.item())])
        
        # MP layer takes predicate scores and propagates them to the adjacent entities
        logits = self.mp(subgraph)
        
        outputs = (logits,)

        if answer is not None:
            # terminate with 0-loss if correct answer is not in the subgraph
            answer_id = answer.item()    
            if answer_id not in entity_ids:
                print("Wrong subgraph selected")
                loss = torch.tensor(0)
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                print(answer_idx)
                print(logits.view(-1, num_labels).shape)
                loss = loss_fct(logits.view(-1, num_labels), answer_idx)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits