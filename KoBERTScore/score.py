import torch
import torch.nn.functional as F


def bert_forwarding(bert_model, input_ids=None, attention_mask=None, output_layer_index=-1):
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
    with torch.no_grad():
        _, _, hidden_states = bert_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True)
    if output_layer_index == 'all':
        return hidden_states
    return hidden_states[output_layer_index]


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


def compute_pairwise_cosine(input_embeds, refer_embeds):
    """
    Args:
        input_embeds (torch.tensor) : (B, K_i, D)
            B : batch size
            K_i : maximum sequence length in `input_embed`
            D : BERT embedding dim
        refer_embeds (torch.tensor) : (B, K_r, D)
            B : batch size
            K_r : maximum sequence length in `refer_embeds`
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

    input_embeds = normalize(input_embeds)
    refer_embeds = normalize(refer_embeds)
    pairwise_cosine = torch.bmm(input_embeds, refer_embeds.permute(0, 2, 1))
    return pairwise_cosine


def bert_score(bert_tokenizer, bert_model, references, hypotheses,
               idf=None, output_layer_index=-1, rescale_base=0):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        references (list of str) : True sentences
        hypotheses (list of str) : Generated sentences
        idf (torch.tensor or None) : IDF weights
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

        >>> refer_sents = ['hello world', 'my name is lovit', 'oh hi', 'where I am', 'where we are going']
        >>> hypoh_sents = ['Hellow words', 'I am lovit', 'oh hello', 'where am I', 'where we go']
        >>> bert_score(bert_tokenizer, bert_model, references, hypotheses)

        $ (tensor([0.6283, 0.7944, 0.8768, 0.6904, 0.7653]),
           tensor([0.5252, 0.8333, 0.8768, 0.6904, 0.8235]),
           tensor([0.5721, 0.8134, 0.8768, 0.6904, 0.7934]))
    """
    # tokenization
    refer_ids, refer_attention_mask, refer_weight_mask = sents_to_tensor(bert_tokenizer, references)
    hypoh_ids, hypoh_attention_mask, hypoh_weight_mask = sents_to_tensor(bert_tokenizer, hypotheses)

    # BERT embedding
    refer_embeds = bert_forwarding(bert_model, refer_ids, refer_attention_mask, output_layer_index)
    hypoh_embeds = bert_forwarding(bert_model, hypoh_ids, hypoh_attention_mask, output_layer_index)

    pairwise_cosine = compute_pairwise_cosine(refer_embeds, hypoh_embeds)
    R_max, _ = pairwise_cosine.max(dim=2)
    P_max, _ = pairwise_cosine.max(dim=1)

    if idf is not None:
        refer_weight_mask = apply_idf(refer_ids, idf)
        hypoh_weight_mask = apply_idf(hypoh_ids, idf)

    R_max = rescaling(R_max, rescale_base)
    P_max = rescaling(P_max, rescale_base)

    R = (R_max * refer_weight_mask).sum(axis=1) / refer_weight_mask.sum(axis=1)
    P = (P_max * hypoh_weight_mask).sum(axis=1) / hypoh_weight_mask.sum(axis=1)
    F = 2 * (R * P) / (R + P)
    return R, P, F


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
    return (scores - base) / (1 - base)
