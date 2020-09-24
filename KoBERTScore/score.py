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
                   [1, 1, 1, 1, 1, 0, 0]]))
    """
    inputs = bert_tokenizer.batch_encode_plus(input_sents, padding=True)
    padded_input_ids = torch.LongTensor(inputs['input_ids'])
    attention_mask = torch.LongTensor(inputs['attention_mask'])
    return padded_input_ids, attention_mask


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

    Examples:
        >>> input_sents = ['Hellow words', 'I am lovit', 'oh hello', 'where am I', 'where we go']
        >>> refer_sents = ['hello world', 'my name is lovit', 'oh hi', 'where I am', 'where we are going']

        >>> input_embeds, input_attention_mask = prepare_bertscore_inputs(tokenizer, encoder, input_sents)
        >>> refer_embeds, refer_attention_mask = prepare_bertscore_inputs(tokenizer, encoder, refer_sents)

        >>> input_embeds.size()          # torch.Size([5, 7, 768])
        >>> input_attention_mask.size()  # torch.Size([5, 7])
    """
    padded_input_ids, attention_mask = sents_to_tensor(bert_tokenizer, input_sents)
    outputs = bert_forwarding(bert_model, padded_input_ids, attention_mask, output_layer_index)
    return outputs, attention_mask


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
