import numpy as np
from collections import Counter
from tqdm import tqdm


def train_idf(bert_tokenizer, references, batch_size=1000, verbose=True):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        references (list of str) : True sentences
        batch_size (int)
        verbose (Boolean)

    Returns:
        idf (numpy.ndarray) : shape = (bert_tokenizer.vocab_size,)
    """
    n_sents = len(references)
    counter = Counter()
    begin_index = list(range(0, n_sents, batch_size))

    if verbose:
        iterator = tqdm(begin_index, total=round(n_sents / batch_size), desc='Train IDF')
    else:
        iterator = begin_index

    for i in iterator:
        encoding = tokenizer.batch_encode_plus(
            references[i: i + batch_size],
            add_special_tokens=False)
        subcounter = Counter(idx for sent in encoding['input_ids'] for idx in sent)
        counter.update(subcounter)

    idf = np.ones(tokenizer.vocab_size)
    indices, df = zip(*counter.items())
    idf[np.array(indices)] += np.array(df)
    idf = 1 / idf
    return idf
