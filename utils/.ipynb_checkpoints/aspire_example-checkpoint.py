from transformers import AutoTokenizer
from ex_aspire_consent import AspireConSent, prepare_abstracts
import torch

# Initialize the tokenizer and model.
hf_model_name = 'allenai/aspire-contextualsentence-singlem-compsci'
aspire_tok = AutoTokenizer.from_pretrained(hf_model_name)
aspire_mv_model = AspireConSent(hf_model_name)


# Example input.
ex_abstracts = [
    {'TITLE': "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific"
              " Document Similarity",
     'ABSTRACT': ["We present a new scientific document similarity model based on "
                  "matching fine-grained aspects of texts.",
                  "To train our model, we exploit a naturally-occurring source of "
                  "supervision: sentences in the full-text of papers that cite multiple "
                  "papers together (co-citations)."]},
    {'TITLE': "CSFCube -- A Test Collection of Computer Science Research Articles for "
              "Faceted Query by Example",
     'ABSTRACT': ["Query by Example is a well-known information retrieval task in which"
                  " a document is chosen by the user as the search query and the goal is "
                  "to retrieve relevant documents from a large collection.",
                  "However, a document often covers multiple aspects of a topic.",
                  "To address this scenario we introduce the task of faceted Query by "
                  "Example in which users can also specify a finer grained aspect in "
                  "addition to the input query document. "]}
]

bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=ex_abstracts,
                                                          pt_lm_tokenizer=aspire_tok)
clsreps, contextual_sent_reps = aspire_mv_model.forward(bert_batch=bert_batch,
                                                        abs_lens=abs_lens,
                                                        sent_tok_idxs=sent_token_idxs)

distance_matrix = torch.cdist(contextual_sent_reps[0], contextual_sent_reps[1], p=2.0)
print(distance_matrix)