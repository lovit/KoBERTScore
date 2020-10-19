import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


def bert_score(bert_tokenizer, bert_model, references, candidates,
               idf=None, output_layer_index=-1, rescale_base=0):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        references (list of str) : True sentences
        candidates (list of str) : Generated sentences
        idf (torch.nn.Embedding or None) : IDF weights
        output_layer_index (int)
            The index of last BERT layer which is used for token embedding
        rescale_base (float) : 0 <= rescale_base < 1
            Adjust (R-BERTScore - base) / (1 - base)

    Returns:
        R (torch.tensor) : R-BERTScore
        P (torch.tensor) : P-BERTScore
        F (torch.tensor) : F-BERTScore

    Examples:
        >>> from transformers import BertModel, BertTokenizer

        >>> model_name = "bert-base-uncased"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> encoder = BertModel.from_pretrained(model_name)

        >>> references = ['hello world', 'my name is lovit', 'oh hi', 'where I am', 'where we are going']
        >>> candidates = ['Hellow words', 'I am lovit', 'oh hello', 'where am I', 'where we go']
        >>> bert_score(bert_tokenizer, bert_model, references, candidates)

        $ (tensor([0.6283, 0.7944, 0.8768, 0.6904, 0.7653]),
           tensor([0.5252, 0.8333, 0.8768, 0.6904, 0.8235]),
           tensor([0.5721, 0.8134, 0.8768, 0.6904, 0.7934]))
    """
    # tokenization
    refer_ids, refer_attention_mask, refer_weight_mask = sents_to_tensor(bert_tokenizer, references)
    candi_ids, candi_attention_mask, candi_weight_mask = sents_to_tensor(bert_tokenizer, candidates)

    # BERT embedding
    refer_embeds = bert_forwarding(bert_model, refer_ids, refer_attention_mask, output_layer_index)
    candi_embeds = bert_forwarding(bert_model, candi_ids, candi_attention_mask, output_layer_index)

    # Compute bert RPF
    R, P, F = compute_RPF(
        refer_embeds, candi_embeds,
        refer_weight_mask, candi_attention_mask,
        refer_ids, candi_ids,
        idf, rescale_base)
    return R, P, F


def sents_to_tensor(bert_tokenizer, input_sents):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        input_sents (list of str)

    Returns:
        padded_input_ids (torch.LongTensor) : (batch, max seq len)
        attention_mask (torch.LongTensor) : (batch, max seq len)
        token_mask (torch.LongTensor) : (batch, max seq len)
            True token is 1 and padded / cls / sep token is 0

    Examples::
        >>> from transformers import BertTokenizer
        >>> model_name = "bert-base-uncased"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> input_sents = ['Hellow words', 'I am lovit', 'oh hello', 'where am I', 'where we go']
        >>> sents_to_tensor(tokenizer, input_sents)
        $ (tensor([[ 101, 7592, 2860, 2616,  102,    0,    0],
                   [ 101, 1045, 2572, 8840, 5737, 2102,  102],
                   [ 101, 2821, 7592,  102,    0,    0,    0],
                   [ 101, 2073, 2572, 1045,  102,    0,    0],
                   [ 101, 2073, 2057, 2175,  102,    0,    0]]),
           tensor([[1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0]]),
           tensor([[0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0]]))
    """
    inputs = bert_tokenizer.batch_encode_plus(input_sents, padding=True)
    padded_input_ids = torch.LongTensor(inputs['input_ids'])
    attention_mask = torch.LongTensor(inputs['attention_mask'])

    zero_mask = torch.zeros(attention_mask.size(), dtype=torch.long)
    token_mask = torch.where(padded_input_ids == bert_tokenizer.cls_token_id, zero_mask, attention_mask)
    token_mask = torch.where(padded_input_ids == bert_tokenizer.sep_token_id, zero_mask, token_mask)
    return padded_input_ids, attention_mask, token_mask


def bert_forwarding(bert_model, input_ids, attention_mask=None, output_layer_index=-1):
    """
    Args:
        bert_model (transformers`s Pretrained models)
        input_ids (torch.LongTensor) : (batch, max seq len)
        attention_mask (torch.LongTensor) : (batch, max seq len)
        output_layer_index (int or str)
            The index of last BERT layer which is used for token embedding
            If type of `output_layer_index` is `str`, it returns hidden states of all layers

    Returns:
        hidden_states (torch.tensor) : (B, K, D) or (n_layers, B, K, D)
            B : batch size
            K : maximum sequence length in `input_ids`
            D : BERT embedding dim
    """
    device = next(bert_model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        _, _, hidden_states = bert_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True)
    if output_layer_index == 'all':
        return [h.cpu() for h in hidden_states]
    return hidden_states[output_layer_index].cpu()


def compute_RPF(refer_embeds, candi_embeds, refer_weight_mask, candi_weight_mask,
                refer_ids=None, candi_ids=None, idf=None, rescale_base=0):
    """
    Args:
        refer_embeds (torch.tensor) : (B, K_i, D)
            B : batch size
            K_i : maximum sequence length in `refer_embeds`
            D : BERT embedding dim
        candi_embeds (torch.tensor) : (B, K_r, D)
            B : batch size
            K_r : maximum sequence length in `candi_embeds`
            D : BERT embedding dim
        refer_weight_mask (torch.tensor) : (batch, max seq len)
            token mask or IDF weight mask
        candi_weight_mask (torch.tensor) : (batch, max seq len)
            token mask or IDF weight mask
        idf (torch.nn.Embedding or None) : IDF weights
        rescale_base (float) : 0 <= rescale_base < 1
            Adjust (R-BERTScore - base) / (1 - base)

    Returns:
        R (torch.tensor) : R-BERTScore
        P (torch.tensor) : P-BERTScore
        F (torch.tensor) : F-BERTScore

    """
    pairwise_cosine = compute_pairwise_cosine(refer_embeds, candi_embeds)
    R_max, _ = pairwise_cosine.max(dim=2)
    P_max, _ = pairwise_cosine.max(dim=1)

    if (idf is not None) and (refer_ids is not None) and (candi_ids is not None):
        refer_weight_mask = apply_idf(refer_ids, idf)
        candi_weight_mask = apply_idf(candi_ids, idf)

    R_max = rescaling(R_max, rescale_base)
    P_max = rescaling(P_max, rescale_base)

    R = (R_max * refer_weight_mask).sum(axis=1) / refer_weight_mask.sum(axis=1)
    P = (P_max * candi_weight_mask).sum(axis=1) / candi_weight_mask.sum(axis=1)
    F = 2 * (R * P) / (R + P)
    return R, P, F


def compute_pairwise_cosine(refer_embeds, candi_embeds):
    """
    Args:
        refer_embeds (torch.tensor) : (B, K_i, D)
            B : batch size
            K_i : maximum sequence length in `refer_embeds`
            D : BERT embedding dim
        candi_embeds (torch.tensor) : (B, K_r, D)
            B : batch size
            K_r : maximum sequence length in `candi_embeds`
            D : BERT embedding dim

    Returns:
        pairwise_cosine (torch.tensor) : (B, K_i, K_r)

    Examples::
        >>> input1 = torch.randn(3, 4, 5)
        >>> input2 = torch.randn(3, 7, 5)
        >>> compute_pairwise_cosine(input1, input2).size()
        $ torch.Size([3, 4, 7])
    """
    def normalize(embeds):
        embeds.div_(torch.norm(embeds, dim=-1).unsqueeze(-1))
        return embeds

    refer_embeds = normalize(refer_embeds)
    candi_embeds = normalize(candi_embeds)
    pairwise_cosine = torch.bmm(refer_embeds, candi_embeds.permute(0, 2, 1))
    return pairwise_cosine


def apply_idf(ids, idf_embed):
    """
    Args:
        ids (torch.tensor) : (batch, max seq len)
        idf_embed (torch.nn.Embedding) : (n vocab, 1)

    Returns:
        embedded (torch.tensor) : (batch, max seq len)

    Examples::
        >>> from torch import nn
        >>> idf_weight = torch.tensor([[0, 0.5, 0.25, 0.3, 5, 3.2]]).t()

        >>> num_vocab = idf_weight.size()[0]
        >>> embed = nn.Embedding(num_vocab, 1, _weight=idf_weight)
        >>> embed.weight.requires_grad = False

        >>> ids = torch.tensor([[0, 1, 2, 3, 2, 3, 0, 0],
        >>>                     [0, 2, 3, 2, 3, 0, 0, 0]])
        >>> apply_idf(ids, idf_embed)
        $ tensor([[0.0000, 0.5000, 0.2500, 0.3000, 0.2500, 0.3000, 0.0000, 0.0000],
                  [0.0000, 0.2500, 0.3000, 0.2500, 0.3000, 0.0000, 0.0000, 0.0000]])
    """
    return idf_embed(ids).squeeze(dim=2)


def rescaling(scores, base):
    """
    Transform `(score - base) / (1 - base)

    For computing `base`, authors use Common Crawl in paper.
    They create 1M candidate-reference pairs by grouping two random sentences.
    Because each pair has very low lexical and semantic overlapping,
    and determine `base` as average BERTScore computed on these sentence pairs.
    - Refer: BERTScore: Evaluating Text Generation with BERT (https://arxiv.org/abs/1904.09675)

    Args:
        scores (float or torch.tensor) : float or (batch, max seq len)
        base (float)

    Returns:
        scores_ (float or torch.tensor) : float or (batch, max seq len)
            Transformed scores
    """
    return (scores - base) / (1 - base)


class BERTScore:
    def __init__(self, model_name_or_path='beomi/kcbert-base', best_layer=-1, idf_path=None, rescale_base=0, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if isinstance(model_name_or_path, tuple):
            self.tokenizer, self.encoder = model_name_or_path
        else:
            self.tokenizer, self.encoder = load_model(model_name_or_path, best_layer)
        self.encoder = self.encoder.to(device)
        self.rescale_base = rescale_base
        self.idf = load_idf(idf_path, self.tokenizer)

    def __call__(self, references, candidates, batch_size=128, retrain_idf=True, verbose=True):
        return self.score(references, candidates, batch_size, retrain_idf, verbose)

    def score(self, references, candidates, batch_size=128, retrain_idf=True, verbose=True):
        n_examples = len(references)
        n_batch = math.ceil(n_examples / batch_size)
        if verbose:
            step_iterator = tqdm(range(n_batch), desc='Calculating BERTScore', total=n_batch)
        else:
            step_iterator = range(n_batch)

        if retrain_idf:
            idf = train_idf(self.tokenizer, references, batch_size=1000, verbose=verbose)
            idf = idf_numpy_to_embed(idf)
        else:
            idf = self.idf

        F = []
        for step in step_iterator:
            b = step * batch_size
            e = min((step + 1) * batch_size, n_examples)
            refer_batch = references[b: e]
            candi_batch = candidates[b: e]

            _, _, F_batch = bert_score(
                self.tokenizer, self.encoder,
                refer_batch, candi_batch,
                idf=self.idf, rescale_base=self.rescale_base)
            F += F_batch.detach().numpy().tolist()
        return F

    def plot_bertscore_detail(self, reference, candidate,
        idf=None, height='auto', width='auto', title=None, return_gridplot=True):
        """
        Args:
            reference (str) : Reference sentence
            candidate (str) : Candidate sentence
            idf (torch.nn.Embedding) : custom IDF weight
                If None, use BERTScore attribute IDF weight
            height (int or str) : Figure height
                If `auto`, it automatically determine the height of figure
                considering the number of tokens in `reference`
            width (int or str) : Figure width
                If `auto`, it automatically determine the height of figure
                considering the number of tokens in `candidate`
            title (str or None) : Figure title
            return_gridplot (Boolean) :
                If True, it returns gridplot formed figure
                Else, it returns tuple of figures (cos, idf)

        Returns:
            figure (bokeh.figure or (bokeh.figure, bokeh.figure)

        Examples::
            Loading BERT manually

                >>> from bokeh.plotting import show
                >>> from KoBERTScore import BERTScore

                >>> bertscore = BERTScore("bert-base-uncased")
                >>> reference = '날씨는 좋고 할일은 많고 어우 연휴 끝났다'
                >>> candidate = '날씨가 좋다 하지만 할일이 많다 일해라 인간'
                >>> p = bertscore.plot_bertscore_detail(reference, candidate)
                >>> show(p)
        """
        if not isinstance(reference, str) or not isinstance(candidate, str):
            raise ValueError('`reference` and `candidate` must be str type')
        if idf is None:
            idf = self.idf

        from .tasks import plot_bertscore_detail
        figure = plot_bertscore_detail(
            reference, candidate, self.tokenizer, self.encoder, idf,
            -1, height, width, title, return_gridplot)
        return figure


MODEL_TO_BEST_LAYER = {
    'beomi/kcbert-base': 4,
    'monologg/kobert': 2,
    'monologg/distilkobert': 12,
    'monologg/koelectra-base-v2-discriminator': 12
}


def load_model(model_name_or_path, best_layer=-1):
    if os.path.exists(model_name_or_path):
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        encoder = BertModel.from_pretrained(model_name_or_path)
    elif model_name_or_path in MODEL_TO_BEST_LAYER:
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        encoder = BertModel.from_pretrained(model_name_or_path)
    else:
        raise ValueError(
            f'Ko-BERTScore uses only {list(MODEL_TO_BEST_LAYER.keys())} or local model'
            'Check `model_name_or_path`')

    if best_layer == -1:
        best_layer = MODEL_TO_BEST_LAYER.get(model_name_or_path, -1)

    if best_layer > 0:
        encoder = truncate_bert_layers(encoder, best_layer)

    print(f'Load {model_name_or_path} with {best_layer} layers')
    return tokenizer, encoder


def truncate_bert_layers(encoder, last_layer):
    encoder.encoder.layer = torch.nn.ModuleList([
        layer for layer in encoder.encoder.layer[:last_layer]
    ])
    encoder.config.num_hidden_layers = last_layer
    return encoder


def load_idf(path, tokenizer):
    if path is None:
        weight = torch.ones((len(tokenizer), 1), dtype=torch.float)
    else:
        with open(path, encoding='utf-8') as f:
            weight = [float(line.strip()) for line in f]
        weight = torch.tensor([weight]).T
    n_vocab = weight.size()[0]

    if len(tokenizer) != n_vocab:
        raise ValueError(
            'The number of vocab in `tokenizer` must be same wigh `idf` size\n'
            f'len(tokenizer)={len(tokenizer)}, len(idf)={n_vocab}')

    idf = torch.nn.Embedding(n_vocab, 1, _weight=weight)
    return idf


def idf_numpy_to_embed(idf_array):
    """
    Args:
        idf_array (numpy.ndarray) : shape=(n_vocab,)

    Returns:
        idf_embed (torch.nn.Embedding) : size=(n_vocab, 1)

    Examples::
        >>> import numpy as np
        >>> idf = np.random.random_sample(10000)
        >>> idf_embed = idf_numpy_to_embed(idf)
        >>> type(idf_embed)
        $ torch.nn.modules.sparse.Embedding
    """
    idf = torch.tensor([idf_array]).T
    idf_embed = torch.nn.Embedding(idf.size()[0], 1, _weight=idf)
    idf_embed.weight.requires_grad = False
    return idf_embed


def train_idf(bert_tokenizer, references, batch_size=1000, verbose=True):
    """
    Train IDF vector with Laplace (add one) smoothing

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
        encoding = bert_tokenizer.batch_encode_plus(
            references[i: i + batch_size],
            add_special_tokens=False)
        subcounter = Counter(idx for sent in encoding['input_ids'] for idx in sent)
        counter.update(subcounter)

    idf = np.ones(bert_tokenizer.vocab_size)
    indices, df = zip(*counter.items())
    idf[np.array(indices)] += np.array(df)
    idf = 1 / idf
    idf[np.array(bert_tokenizer.all_special_ids, dtype=np.int)] = 0
    return idf
