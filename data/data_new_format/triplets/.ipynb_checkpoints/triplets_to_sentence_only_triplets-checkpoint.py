#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


import sys
sys.path.append('/cs/labs/tomhope/taltatal/aspire')       
import utils.envsetup
import pickle
import tqdm
import torch


# In[ ]:


from transformers import AutoModel, AutoTokenizer

aspire_sent = AutoModel.from_pretrained('allenai/aspire-sentence-embedder')
aspire_tok = AutoTokenizer.from_pretrained('allenai/aspire-sentence-embedder')

def apply_sent_bert(sents):
    inputs = aspire_tok(sents, padding=True, truncation=True, return_tensors="pt", max_length=512)

    result = aspire_sent(**inputs)

    clsrep = result.last_hidden_state[:,0,:]
    
    return clsrep


# In[ ]:


def remove_irrelevant_sentences_from_data(sentences):
    # Created in order to assure quality of train couples
    cond = lambda s: True # len(s) > 8 and len(s.split(" ")) > 1
    return [s for s in sentences if cond(s)]


# In[ ]:


def triplets_to_sentence_only(triplets):
    triplets_sentence_only = []
    for triplet in tqdm.tqdm(triplets):
        # generate embeddings
        topic, pos, neg = triplet[0]['ABSTRACT'], triplet[1]['ABSTRACT'], triplet[2]['ABSTRACT']
        topic = remove_irrelevant_sentences_from_data(topic)
        pos = remove_irrelevant_sentences_from_data(pos)
        neg = remove_irrelevant_sentences_from_data(neg)

        # IMPORTANT!!! I CHANGED THE CO-CITATION-CONTEXT INTO "TITLE + CO-CITATION-CONTEXT"
        co_citation_context = triplet[0]["TITLE"] + " " + triplet[0]["CO-CITATION-CONTEXT"]
        topic_embed, pos_embed, neg_embed, context_embed = apply_sent_bert(topic), apply_sent_bert(pos), apply_sent_bert(neg), apply_sent_bert(co_citation_context)

        # create sentence-only triplet
        triplet_sentence_only = []
        for sentences, embedding in zip([topic, pos, neg], [topic_embed, pos_embed, neg_embed]):
            distance_embedding_context = torch.squeeze(torch.cdist(embedding, context_embed, p=2.0), 1)
            argmin = torch.argmin(distance_embedding_context)
            triplet_sentence_only.append(sentences[argmin])
        triplets_sentence_only.append(tuple(triplet_sentence_only))
    return triplets_sentence_only


# # test

# In[ ]:


with open("test_triplets.pkl", "rb") as file:
    test_triplets = pickle.load(file)


# In[ ]:


test_triplets_sentence_only = triplets_to_sentence_only(test_triplets)


# In[ ]:


with open("test_triplets_sentence_only.pkl", "wb") as file:
    pickle.dump(test_triplets_sentence_only, file)


# # train

# In[ ]:


# with open("train_triplets.pkl", "rb") as file:
#     train_triplets = pickle.load(file)


# In[ ]:


# train_triplets_sentence_only = triplets_to_sentence_only(train_triplets)


# In[ ]:


# with open("train_triplets_sentence_only.pkl", "wb") as file:
#     pickle.dump(train_triplets_sentence_only, file)

