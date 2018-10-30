from nltk.tokenize import sent_tokenize, TweetTokenizer
from string import punctuation
from os import scandir
from operator import itemgetter
from collections import defaultdict

tokenizer = TweetTokenizer()

def main():
    token_docid, doc_ids = get_token_doc_id_pairs('/Users/i322053/fmi/information_retrieval_fmi/data/mini_newsgroups/rec.autos/')
    sorted_token_docid = sorted(token_docid, key=itemgetter(0))
    tokens_in_doc = merge_token_in_doc(sorted_token_docid)
    dictionary = defaultdict(lambda: (0, 0))  # term : doc_freq, tot freq
    postings = defaultdict(lambda: [])  # term: doc_ids, doc_freq

    for token, doc_id, doc_freq in tokens_in_doc:
        dictionary[token] = (dictionary[token][0] + 1, dictionary[token][0] + doc_freq)

    # usually implemented as linked lists
    for token, doc_id, doc_freq in tokens_in_doc:
        postings[token].append((doc_id, doc_freq))

    doc_id = and_query(postings, 'living', 'dead')
    print(doc_ids[doc_id[0]])


def preprocess_document(content):
    """
    Returns a list of tokens for a document's content.
    Tokens should not contain punctuation and should be lower-cased.
    """
    sentences = sent_tokenize(content)
    tokens = []
    for _sent in sentences:
        sent_tokens = tokenizer.tokenize(_sent)
        sent_tokens = [_tok.lower() for _tok in sent_tokens if _tok not in punctuation]
        tokens += sent_tokens
    return tokens


def get_token_doc_id_pairs(category_dir):
    """
    Iteratively goes through the documents in the category_dir and constructs/returns:
    1. A list of (token, doc_id) tuples
    2. A dictionary of doc_id:doc_name
    """
    token_docids = []
    doc_ids = {}
    for i, entry in enumerate(scandir(category_dir)):
        if entry.is_file():
            doc_ids[i] = entry
            tokens = preprocess_document(open(entry, encoding='ISO-8859-1').read())
            for token in tokens:
                token_docids.append((token, i))
    return token_docids, doc_ids


def merge_token_in_doc(sorted_token_docids):
    """
    Returns a list of (token, doc_id, term_freq) tuples from a sorted list of (token, doc_id) list,
    where if a token appears n times in a doc_id, we merge it in a tuple (toke, doc_id, n).
    """
    token_doc_freq = []
    token_doc_freq_map = {}
    for (token, doc_id) in sorted_token_docids:
        if (token, doc_id) in token_doc_freq_map:
            token_doc_freq_map[(token, doc_id)] = token_doc_freq_map[(token, doc_id)] + 1
        else:
            token_doc_freq_map[(token, doc_id)] = 1

    for (token, doc_id), freq in token_doc_freq_map.items():
        token_doc_freq.append((token, doc_id, freq))
    return token_doc_freq

def and_query(postings, word1, word2):
    """
    merging postings lists of two words
    """
    postings_word1 = postings[word1]
    postings_word2 = postings[word2]

    documents_results = []

    postings_ind1, postings_ind2 = 0, 0
    while postings_ind1 < len(postings_word1) and postings_ind2 < len(postings_word2):
        doc_id1, doc_id2 = postings_word1[postings_ind1][0], postings_word2[postings_ind2][0]
        if doc_id1 == doc_id2:
            documents_results.append(postings_word1[postings_ind1][0])
            postings_ind1 += 1
            postings_ind2 += 1
        elif doc_id1 < doc_id2:
            postings_ind1 += 1
        elif doc_id1 > doc_id2:
            postings_ind2 += 1
    return documents_results

if __name__ == '__main__':
    main()
