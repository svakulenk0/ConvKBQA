# load graph
import json
from collections import Counter, defaultdict
import numpy as np

from hdt import HDTDocument, TripleComponentRole

from settings import *
from predicates import properties


hdt_file = 'wikidata2018_09_11.hdt'
kg = HDTDocument(hdt_path+hdt_file)
namespace = 'predef-wikidata2018-09-all'
PREFIX_E = 'http://www.wikidata.org/entity/'
PREFIX_P = 'http://www.wikidata.org/prop/'

# prepare to retrieve all adjacent nodes including literals
predicates_ids = []
kg.configure_hops(1, predicates_ids, namespace, True, False)

# load all predicate labels
relationid2label = {}
for p in properties['results']['bindings']:
    _id = p['property']['value'].split('/')[-1]
    label = p['propertyLabel']['value']
    relationid2label[_id] = label

# model init
import torch
from transformers import DistilBertTokenizer, DistilBertConfig

from MPBert_sampler_model import MessagePassingHDTBert
from utils import adj

# fix random seed for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load all predicate labels
from predicates import properties

relationid2label = {}
for p in properties['results']['bindings']:
    _id = p['property']['value'].split('/')[-1]
#     print(_id)
    label = p['propertyLabel']['value']
    relationid2label[_id] = label
    
# all unique predicate labels
all_predicate_labels = list(relationid2label.values())
all_predicate_ids = [kg.string_to_global_id(PREFIX_P + p, TripleComponentRole.PREDICATE) for p in list(relationid2label.keys())]
assert len(all_predicate_labels) == len(all_predicate_ids)
# print(all_predicate_ids[0])

print("Graph loaded")

# model init
import torch

from transformers import DistilBertTokenizer, DistilBertConfig
from MPBert_sampler_model import MessagePassingHDTBert

DEVICE = 'cuda'

# model configuration
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
config = DistilBertConfig.from_pretrained(model_name, num_labels=1)

E_BEAM = 10
P_BEAM = 100

model = MessagePassingHDTBert(config, topk_entities=E_BEAM, topk_predicates=P_BEAM)

for param in model.bert.parameters():
    param.requires_grad = False


if DEVICE == 'cuda':
    device = torch.device("cuda")
    # run model on the GPU
    model.cuda()
else:
    # use CPU to train the model
    device = torch.device("cpu")

print("Model loaded to", DEVICE)

# check how many times an answer to the question fall into the initial (seed) subgraph separately for each order in the question sequence
NSAMPLES = 200
max_triples = 50000
offset = 0

# dataset setup
train_conversations_path = '../data/train_set/train_set_movies.json'
dev_conversations_path = '../data/dev_set/dev_set_movies.json'


# collect only samples where the answer is entity and it is adjacent to the seed entity
train_dataset = []

graph_sizes = []
max_n_edges = 2409 # max size of the graph allowed in the number of edges


rdfsLabelURI='http://www.w3.org/2000/01/rdf-schema#label'

def lookup_entity_labels(entity_ids):
    # prepare mapping tensors with entity labels and ids
    entity_labels, s_entity_ids = [], []
    for i, e_id in enumerate(entity_ids):
        e_uri = kg.global_id_to_string(e_id, TripleComponentRole.OBJECT)
        (triples, cardinality) = kg.search_triples(e_uri, rdfsLabelURI, "")
        if cardinality > 0:
            label = triples.next()[2]
            # strip language marker
            label = label.split('"')[1]
            entity_labels.append(label)
            s_entity_ids.append(e_id)

    assert len(entity_labels) == len(s_entity_ids)
    return entity_labels, s_entity_ids


def prepare_dataset(train_conversations_path, n_limit=NSAMPLES):
    with open(train_conversations_path, "r") as data:
            conversations = json.load(data)
    print("%d conversations loaded"%len(conversations))

    # consider a sample of the dataset
    if n_limit:
        conversations = conversations[:n_limit]

    n_entities = []
    n_edges = []

    train_dataset = []

    for conversation in conversations[:NSAMPLES]:
        # store history of the current conversation
        dialogue_history = []
        for i in range(len(conversation['questions'][:1])):

            question = conversation['questions'][i]['question']
            answer = conversation['questions'][i]['answer']
            # use oracle for the correct initial entity
            seed_entity = conversation['seed_entity'].split('/')[-1]
            seed_entity_id = kg.string_to_global_id(PREFIX_E+seed_entity, TripleComponentRole.OBJECT)

            # retrieve all adjacent nodes including literals
            subgraph = kg.compute_hops([seed_entity_id], max_triples, offset)
            entity_ids, predicate_ids, adjacencies = subgraph

            assert len(predicate_ids) == len(adjacencies)
        #         print("conversation")

            # answer literal
            answer_label = answer
            # consider only answers which are entities
            if ('www.wikidata.org' in answer):
                answer_id = kg.string_to_global_id(PREFIX_E+answer.split('/')[-1], TripleComponentRole.OBJECT)
                in_subgraph = answer_id in entity_ids

                # look up answer entity label
                a_uri = PREFIX_E+answer.split('/')[-1]
                (triples, cardinality) = kg.search_triples(a_uri, rdfsLabelURI, "")
                if cardinality > 0:
                    answer_label = triples.next()[2]
                    # strip language marker
                    answer_label = answer_label.split('"')[1]

                # retain samples with answer outside the seed subgraph
                p_input_ids = []
                p_token_type_ids = []
                p_attention_masks = []

                # prepare input of questions concatenated with all relation labels in the KG as candidates
                # trim predicates
                for p_label in all_predicate_labels:
                    # encode a text pair of the question with a predicate label
                    encoded_dict = tokenizer.encode_plus(question, p_label,
                                                         add_special_tokens=True,
                                                         max_length=20,
                                                         pad_to_max_length=True,
                                                         return_attention_mask=True)
                    p_input_ids.append(encoded_dict['input_ids'])
#                     p_token_type_ids.append(encoded_dict['token_type_ids'])
                    p_attention_masks.append(encoded_dict['attention_mask'])


                # prepare input of questions concatenated with node labels as candidates: get labels for all candidate entities in the seed subgraph
                entity_labels, entity_ids = lookup_entity_labels(entity_ids)
                # create a batch of samples for each entity label separately
                e_input_ids = []
                e_token_type_ids = []
                e_attention_masks = []
                for e_label in entity_labels:
                    # encode a text pair of the question with a predicate label
                    encoded_dict = tokenizer.encode_plus([question]+dialogue_history[::-1], e_label,
                                                         add_special_tokens=True,
                                                         max_length=64,
                                                         pad_to_max_length=True,
                                                         return_attention_mask=True)
                    e_input_ids.append(encoded_dict['input_ids'])
#                     e_token_type_ids.append(encoded_dict['token_type_ids'])
                    e_attention_masks.append(encoded_dict['attention_mask'])
                assert len(e_input_ids) == len(entity_ids)
                first_question = None
                if i == 0 and in_subgraph:
#                     print('first question')
                    first_question = torch.tensor([seed_entity_id])
                    train_dataset.append([[torch.tensor(e_input_ids), torch.tensor(e_token_type_ids),
                                           torch.tensor(e_attention_masks), torch.tensor(entity_ids)],
                                          [torch.tensor(p_input_ids), torch.tensor(p_token_type_ids),
                                           torch.tensor(p_attention_masks), torch.tensor(all_predicate_ids)],
                                           torch.tensor([answer_id]), first_question])
            # carry over history to the next dialogue turn
            dialogue_history.extend([question, answer_label])

    del entity_ids, predicate_ids, adjacencies

    print("Compiled dataset with %d samples" % len(train_dataset))
    return train_dataset


train_dataset = prepare_dataset(train_conversations_path)
valid_dataset = prepare_dataset(dev_conversations_path)

# remove everything from memory but model and tensors for training/validaton
kg.remove()
del kg

print("Dataset loaded")

# train model (matching nodes and relations with a Transformer with subgraph sampling)
n_batches = 1000

import random
import numpy as np
from functools import reduce
import sys
import os
import gc
import psutil

# training setup
from transformers import get_linear_schedule_with_warmup, AdamW

epochs = 4
total_steps = len(train_dataset) * epochs

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                 )
# learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

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
    
#     print("Started epoch")
#     memoryStats()
    
    # for each sample of training data input as a batch of size 1
    n_losses = 0
    for step, batch in enumerate(train_dataset[:n_batches]):
#         print(step)
        
        e_inputs = [tensor.to(device) for tensor in batch[0]]
        p_inputs = [tensor.to(device) for tensor in batch[1]]
        labels = batch[2].to(device)
        first_question = None
        if batch[3]:
            first_question = batch[3].to(device)
        
        model.zero_grad()
        
#         print("Sample ready")
#         memoryStats()
        
        # forward pass
        loss, logits, entity_ids = model(e_inputs,
                                         p_inputs,
                                         labels,
                                         first_question)
        
        del e_inputs, p_inputs, labels, logits
        
#         print(loss.item())
        # accumulate the training loss over all of the batches
        
        
#         print("Forward pass complete")
#         memoryStats()

        # clean up
        gc.collect()
        torch.cuda.empty_cache()

        if not loss == None:
            total_train_loss += float(loss.item())
            print("Correct subgraph selected")
#             memoryStats()
            # backward pass
            loss.backward()
        else:
            total_train_loss += 1
#             print("Backprop complete")
#             memoryStats()
        
        # clip gradient to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update parameters
        optimizer.step()
        scheduler.step()
        
        # clean up
        gc.collect()
        torch.cuda.empty_cache() 
        
    # training epoch is over here
#     print("Training epoch complete")
#     memoryStats()
    
    # calculate average loss over all the batches
    avg_train_loss = total_train_loss / len(train_dataset)
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # put the model in evaluation mode
    model.eval()

    total_eval_loss = 0

    # evaluate data for one epoch
    n_losses = 0
    for step, batch in enumerate(valid_dataset[:n_batches]):
#         print(step)
        
        e_inputs = [tensor.to(device) for tensor in batch[0]]
        p_inputs = [tensor.to(device) for tensor in batch[1]]
        labels = batch[2].to(device)
        first_question = None
        if batch[3]:
            first_question = batch[3].to(device)
        
#         print("Sample ready")
#         memoryStats()

        with torch.no_grad():
            # forward pass
            loss, logits, entity_ids = model(e_inputs,
                                             p_inputs,
                                             labels,
                                             first_question)
            
            if not loss == None:
                # accumulate validation loss
                total_eval_loss += loss.item()
                n_losses += 1
                print("Correct subgraph selected")
            else:
                total_eval_loss += 1
            
#             print("Forward pass complete")
#             memoryStats()
        
        # clean up
        gc.collect()
        torch.cuda.empty_cache()
    
#     print("Validation epoch complete")
#     memoryStats()

    avg_val_loss = total_eval_loss / len(valid_dataset)
    print("Average validation Loss: {0:.2f}".format(avg_val_loss))


model_path = './models/mpbert_%d/'
version = 0
output_dir = model_path % (version + epochs)

print("Saving model to %s" % output_dir)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)