from collections import namedtuple

import utils.envsetup
from data.pmc_iterable import PMCIterable
from data.topic_iterable import TopicIterable
import pickle
import torch
from torch.utils.data import DataLoader
from utils.doc_abstract_to_sentence_list_transform import DocAbstractToSentenceListTransform

from transformers import AutoTokenizer
# from aspire.utils.ex_aspire_consent import AspireConSent, prepare_abstracts
from utils.ex_aspire_consent_multimatch import AspireConSent, AllPairMaskedWasserstein
from utils.ex_aspire_consent_multimatch import prepare_abstracts

import tqdm
import torch.nn.functional as F

# topics = [x['ID'] for x in TopicIterable(format='aspire', topic_file_name='data/data_old_format/topics.pkl')]
# topics.sort()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# huggingface_model_name = 'allenai/aspire-contextualsentence-singlem-compsci' # single match
huggingface_model_name = 'allenai/aspire-contextualsentence-multim-compsci'  # multi match
aspire_tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
aspire_mv_model = AspireConSent(huggingface_model_name).to(device)
# Empty dict of hyper params will force class to use defaults.
ot_distance = AllPairMaskedWasserstein({})


def apply_model(docs, tokenizer, model):
    bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=docs,
                                                              pt_lm_tokenizer=tokenizer)
    # move batch to device, bert_batch is a dict
    for k, v in bert_batch.items():
        bert_batch[k] = v.to(device) if type(v) == torch.Tensor else v
    # abs_lens is a list
    abs_lens = torch.tensor(abs_lens, dtype=torch.long, device=device)

    clsreps, contextual_sent_reps = model.forward(bert_batch=bert_batch,
                                                  abs_lens=abs_lens,
                                                  sent_tok_idxs=sent_token_idxs)
    return abs_lens, contextual_sent_reps


def pad_to_same_size_along_axis(tensor_list, axis=0):
    max_size = max([x.shape[axis] for x in tensor_list])
    padded_list = []
    for tensor in tensor_list:
        if tensor.shape[axis] < max_size:
            pad_size = max_size - tensor.shape[axis]
            padded_list.append(F.pad(input=tensor, pad=(0, 0, 0, pad_size, 0, 0), mode='constant', value=0))
        else:
            padded_list.append(tensor)
    return padded_list

print("loading data")
pmc_ids = pickle.load(open('data/data_old_format/pmc_ids.pkl', 'rb'))

topics_dataloader = DataLoader(TopicIterable(format='aspire', topic_file_name='data/data_old_format/topics.pkl',
                                             transform=DocAbstractToSentenceListTransform()), batch_size=1,
                               collate_fn=lambda x: x)

topics_dataloader = sorted([x for x in topics_dataloader], key=lambda x: int(x[0]['ID']))

print("starting")
topic_embeddings = []
topic_abs_lens = []
for topic_batch in tqdm.tqdm(topics_dataloader):
    # move topic_batch to device, but avoid non tensors. its a list of dicts
    topic_batch = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()} for x in topic_batch]

    abs_lens, topic_embedding = apply_model(topic_batch, aspire_tokenizer, aspire_mv_model)
    topic_embeddings.append(topic_embedding)
    topic_abs_lens.append(abs_lens)
topic_embeddings = pad_to_same_size_along_axis(topic_embeddings, axis=1)
topic_embeddings_tensor = torch.cat(topic_embeddings, dim=0)

rankings = {x[0]['ID']: list() for x in topics_dataloader}

for article_batch in tqdm.tqdm(DataLoader(PMCIterable(labeled_ids_or_filename=pmc_ids, format='aspire',
                                                      transform=DocAbstractToSentenceListTransform()), batch_size=1,
                                          collate_fn=lambda x: x)):
    article_abs_lens, article_embedding = apply_model(article_batch, aspire_tokenizer, aspire_mv_model)
    # distance_matrix = torch.cdist(article_embedding[0], topic_embeddings_tensor, p=2.0)
    query_embeds = article_embedding

    for topic_abs_len, topic_embedding, i in zip(topic_abs_lens, topic_embeddings, range(len(topic_abs_lens))):
        cand_embeds = topic_embedding
        rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
        qt = rep_len_tup(embed=query_embeds.permute(0, 2, 1), abs_lens=[article_abs_lens[0]])
        ct = rep_len_tup(embed=cand_embeds.permute(0, 2, 1), abs_lens=[topic_abs_len[0]])
        wd, intermediate_items = ot_distance.compute_distance(query=qt, cand=ct, return_pair_sims=True)
        rankings[topics_dataloader[i][0]['ID']].append((article_batch[0]['ID'], wd))
        break

import pickle

for k in rankings.keys():
    [k].sort(key=lambda x: x[1])
    pickle.dump(rankings, open('data/data_new_format/aspire_only/{}.pkl'.format(k), 'wb'))

    # transport_plan = intermediate_items[3].data.numpy()[0, :article_abs_lens[0], :topic_abs_lens[0]]
    # print(transport_plan.shape)
    # # Print the sentences and plot the optimal transport plan for the pair of abstracts.
    # print('\n'.join([f'{i}: {s}' for i, s in enumerate(topics_dataloader[0][0]['ABSTRACT'])]))
    # print('')
    # print('\n'.join([f'{i}: {s}' for i, s in enumerate(article_batch[0]['ABSTRACT'])]))
    # h = sns.heatmap(transport_plan, linewidths=.7, cmap='Blues')
    # h.set(xlabel='Candidate', ylabel='Query')
    # h.tick_params(labelsize=5)
    # plt.show()
