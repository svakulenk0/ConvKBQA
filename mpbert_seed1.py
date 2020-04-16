#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Apr 16, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

1-hop MPBert without node selection and graph expansion
'''
import os
import json
from collections import Counter, defaultdict

import random
import numpy as np

import torch
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup, AdamW

from hdt import HDTDocument, TripleComponentRole

from MPBert_model import MessagePassingBert
from utils import *
from settings import *
from predicates import properties

print("Started..")

SUB_SAMPLE = None

# training setup
train_conversations_path = './data/train_set/train_set_ALL.json'
dev_conversations_path = './data/dev_set/dev_set_ALL.json'

model_name = 'bert-base-uncased'
epochs = 4

hdt_file = 'wikidata2018_09_11.hdt'
namespace = 'predef-wikidata2018-09-all'
PREFIX_E = 'http://www.wikidata.org/entity/'

output_dir = './saved_models/1_1stquestion'


def lookup_predicate_labels(kg, predicate_ids):
    p_labels_map = defaultdict(list)
    for p_id in predicate_ids:
        p_uri = kg.global_id_to_string(p_id, TripleComponentRole.PREDICATE)
        label = p_uri.split('/')[-1]
        if label in relationid2label:
            label = relationid2label[label]
        else:
            label = label.split('#')[-1]
        p_labels_map[label].append(p_id)
    return p_labels_map


def check_answer_in_subgraph(kg, conversation, entity_ids):
    answer1 = conversation['questions'][0]['answer']
    # consider only answers which are entities
    if ('www.wikidata.org' in answer1):
        answer1_id = kg.string_to_global_id(PREFIX_E+answer1.split('/')[-1], TripleComponentRole.OBJECT)
        in_subgraph = answer1_id in entity_ids
        # consider only answer entities that are in the subgraph
        if in_subgraph:
            answer1_idx = entity_ids.index(answer1_id)
            return answer1_idx


def prepare_dataset(conversations_path, n_limit=None):
    with open(conversations_path, "r") as data:
        conversations = json.load(data)
    print("%d conversations loaded"%len(conversations))
    
    max_triples = 50000000
    offset = 0

    # collect only samples where the answer is entity and it is adjacent to the seed entity
    train_dataset = []

    graph_sizes = []
    max_n_edges = 2409 # max size of the graph allowed in the number of edges
    if n_limit:
        conversations = conversations[:n_limit]
    for conversation in conversations:
        question1 = conversation['questions'][0]['question']
        # use oracle for the correct initial entity
        seed_entity = conversation['seed_entity'].split('/')[-1]
        seed_entity_id = kg.string_to_global_id(PREFIX_E+seed_entity, TripleComponentRole.OBJECT)

        # retrieve all adjacent nodes including literals
        subgraph = kg.compute_hops([seed_entity_id], max_triples, offset)
        entity_ids, predicate_ids, adjacencies = subgraph
        num_entities = len(entity_ids)
        num_relations = len(predicate_ids)
        
        if not len(entity_ids) <= num_entities:
            continue  # skip samples with large subgraphs
        assert len(predicate_ids) == len(adjacencies)
        
        # check that the answer is in the subgraph
        answer1_idx = check_answer_in_subgraph(kg, conversation, entity_ids)
        if answer1_idx:
            # activate seed entity
            entities = torch.zeros(num_entities, 1)
            entities[[entity_ids.index(seed_entity_id)]] = 1
            
            # get labels for all candidate predicates
            p_labels_map = lookup_predicate_labels(kg, predicate_ids)

            # create a batch of samples for each predicate label separately
            input_ids = []
            attention_masks = []
            token_type_ids = []
            A = []

            for p_label, p_ids in p_labels_map.items():

                # encode a text pair of the question with a predicate label
                encoded_dict = tokenizer.encode_plus(question1, p_label,
                                                     add_special_tokens=True,
                                                     max_length=64,
                                                     pad_to_max_length=True,
                                                     return_attention_mask=True,
                                                     return_token_type_ids=True)
                input_ids.append(encoded_dict['input_ids'])
                token_type_ids.append(encoded_dict['token_type_ids'])
                attention_masks.append(encoded_dict['attention_mask'])

                # get adjacencies only for the predicates sharing the same label
                selected_adjacencies = []
                for p_id in p_ids:
                    p_id_idx = predicate_ids.index(p_id)
                    # add all edges together
                    for edge in adjacencies[p_id_idx]:
                        if edge not in selected_adjacencies:
                            selected_adjacencies.append(edge)
                A.append(selected_adjacencies)

            # create a single graph per example for all predicates
            indices, relation_mask = adj(A, num_entities, num_relations)

            train_dataset.append([torch.tensor(input_ids),
                                  torch.tensor(token_type_ids),
                                  torch.tensor(attention_masks),
                                  [indices, relation_mask, entities],
                                  torch.tensor([answer1_idx])])

    print("Compiled dataset with %d samples" % len(train_dataset))
    return train_dataset


def run_inference(model, dataset):
    # put model in evaluation mode
    model.eval()
    
    # TODO add MRR
    p1s = []  # measure accuracy of the top answer: P@1
    for batch in dataset:
        b_input_ids = batch[0].to(device)
        b_token_mask = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_graphs = [tensor.to(device) for tensor in batch[3]]
        b_labels = batch[4].to(device)
        
        with torch.no_grad():
            # forward pass
            loss, logits = model(b_input_ids,
                                 b_graphs,
                                 token_type_ids=b_token_mask,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            predicted_label = np.argmax(logits.numpy()).flatten()[0]
            true_label = b_labels.numpy()[0]
            p1 = int(predicted_label == true_label)
            p1s.append(p1)
    
    return p1s


# model init
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=1)
model = MessagePassingBert(config)

# KG init
kg = HDTDocument(hdt_path+hdt_file)
predicates_ids = []
kg.configure_hops(1, predicates_ids, namespace, True, False)

# load all predicate labels
relationid2label = {}
for p in properties['results']['bindings']:
    _id = p['property']['value'].split('/')[-1]
    label = p['propertyLabel']['value']
    relationid2label[_id] = label

train_dataset = prepare_dataset(train_conversations_path, n_limit=SUB_SAMPLE)
valid_dataset = prepare_dataset(dev_conversations_path, n_limit=SUB_SAMPLE)

total_steps = len(train_dataset) * epochs

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                 )
# learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# use CPU to train the model
device = torch.device("cpu")

print("%d training examples"%(len(train_dataset)))
print("%d validation examples"%(len(valid_dataset)))

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    
    # reset the total loss for this epoch
    total_train_loss = 0
    
    # put the model into training mode
    model.train()
    
    # for each sample of training data input as a batch of size 1
    for step, batch in enumerate(train_dataset):
        b_input_ids = batch[0].to(device)
        b_token_mask = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_graphs = [tensor.to(device) for tensor in batch[3]]
        b_labels = batch[4].to(device)
        model.zero_grad()
        # forward pass
        loss, logits = model(b_input_ids,
                             b_graphs,
                             token_type_ids=b_token_mask,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        # accumulate the training loss over all of the batches
        total_train_loss += loss.item()

        # backward pass
        loss.backward()
        
        # clip gradient to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update parameters
        optimizer.step()
        scheduler.step()
    
    # training epoch is over here
    
    # calculate average loss over all the batches
    avg_train_loss = total_train_loss / len(train_dataset) 
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")
    
    # put the model in evaluation mode
    model.eval()
    
    total_eval_loss = 0
        
    # evaluate data for one epoch
    for step, batch in enumerate(valid_dataset):
        
        b_input_ids = batch[0].to(device)
        b_token_mask = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_graphs = [tensor.to(device) for tensor in batch[3]]
        b_labels = batch[4].to(device)
        
        with torch.no_grad():
            # forward pass
            loss, logits = model(b_input_ids,
                                 b_graphs,
                                 token_type_ids=b_token_mask,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            # accumulate validation loss
            total_eval_loss += loss.item()
    
    avg_val_loss = total_eval_loss / len(valid_dataset)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


        
p1s = run_inference(model, train_dataset)
print("Train set P@1: %.2f" % np.mean(p1s))

p1s = run_inference(model, valid_dataset)
print("Dev set P@1: %.2f" % np.mean(p1s))
