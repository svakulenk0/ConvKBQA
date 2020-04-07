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


class MPLayer(nn.Module):
    def __init__(self, num_entities):
        super(MPLayer, self).__init__()
        self.num_entities = num_entities

    def forward(self, p, A):
        '''
        Inputs:
            *p*: predicate scores from Transformer
            *adjacencies*: graph edges
        Outputs:
            *y*: answer activations
        '''
#         print(A.shape)
#         print(p)
        
        # propagate score from the Transformer
        _A = A * p
        
        # create a vector to propagate across all available edges, i.e., from all entities, with a score 1
        x = torch.ones(self.num_entities, 1, requires_grad=True)

#         print(x.shape)
        # MP step: propagates entity activations to the adjacent nodes
        y = torch.sparse.mm(_A, x)
#         print(y.shape)
        
        return y


class MessagePassingBert(BertPreTrainedModel):
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
    """  # noqa: ignore flake8"

    def __init__(self, config, num_entities, weight=None, mp_layer=True):
        super(MessagePassingBert, self).__init__(config)
        self.mp_layer = mp_layer
        if self.mp_layer:
            self.num_labels = num_entities
        else:
            self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # the predicted score is then propagated via a message-passing layer
        self.mp = MPLayer(num_entities)
        # the output is a vector with the score distribution over the input entities
        self.weight = weight

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        adjacencies=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # Complains if input_embeds is kept

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        predicate_logits = self.classifier(pooled_output)
        if self.mp_layer == True:
            # MP layer takes predicate scores and propagates them to the adjacent entities
            logits = self.mp(predicate_logits.item(), adjacencies)
        else:
            logits = predicate_logits
        
        
#         print(logits.shape)
#         print(logits.view(-1, self.num_labels).shape)
#         print(labels.view(-1).shape)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)