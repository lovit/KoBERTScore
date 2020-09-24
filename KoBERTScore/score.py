import torch
import torch.nn.functional as F


def bert_forwarding(bert_model, input_ids=None, attention_mask=None, output_layer_index=-1):
    """
    Args:
        bert_model (transformers`s Pretrained models)
        input_ids (torch.LongTensor) : (batch, max seq len)
        attention_mask (torch.LongTensor) : (batch, max seq len)
        output_layer_index (int)
            The index of last BERT layer which is used for token embedding

    Returns:
        hidden_states (torch.tensor) : (B, K, D)
            B : batch size
            K : maximum sequence length in `input_ids`
            D : BERT embedding dim
    """
    with torch.no_grad():
        _, _, hidden_states = bert_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True)
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


def prepare_bertscore_inputs(bert_tokenizer, bert_model, input_sents, output_layer_index=-1):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        input_sents (list of str)
        output_layer_index (int)
            The index of last BERT layer which is used for token embedding

    Returns:
        outputs (torch.tensor) : (batch, max seq len, bert embed dim)
        attention_mask (torch.tensor) : (batch, max seq len)
        token_mask (torch.LongTensor) : (batch, max seq len)
            True token is 1 and padded / cls / sep token is 0

    Examples:
        >>> refer_sents = ['hello world', 'my name is lovit', 'oh hi', 'where I am', 'where we are going']
        >>> hypoh_sents = ['Hellow words', 'I am lovit', 'oh hello', 'where am I', 'where we go']

        >>> refer_embeds, refer_attention_mask, refer_token_mask = prepare_bertscore_inputs(
        >>>     tokenizer, encoder, refer_sents)
        >>> hypoh_embeds, hypoh_attention_mask, hypoh_token_mask = prepare_bertscore_inputs(
        >>>     tokenizer, encoder, input_sents)

        >>> refer_embeds.size()          # torch.Size([5, 8, 768])
        >>> refer_attention_mask.size()  # torch.Size([5, 8])
    """
    padded_input_ids, attention_mask, token_mask = sents_to_tensor(bert_tokenizer, input_sents)
    outputs = bert_forwarding(bert_model, padded_input_ids, attention_mask, output_layer_index)
    return outputs, attention_mask, token_mask


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
