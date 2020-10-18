import math
import numpy as np
import torch
from bokeh.layouts import gridplot
from tqdm import tqdm

from .score import sents_to_tensor, bert_forwarding
from .utils import correlation, lineplot


def find_best_layer(bert_tokenizer, bert_model, references, candidates, qualities,
                    idf=None, rescale_base=0, model_name=None, batch_size=128, draw_plot=True):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        references (list of str) : True sentences
        candidates (list of str) : Generated sentences
        qualities (list of float) : True qualities between (reference, candidate)
        idf (torch.nn.Embedding or None) : IDF weights
        rescale_base (float) : 0 <= rescale_base < 1
            Adjust (R-BERTScore - base) / (1 - base)
        batch_size (int) : Batch size, default = 128
        draw_plot (Boolean) : If True, it returns bokeh plots

    Returns:
        best_layer (int) : Best-layer index
        information (dict) : consists with `R`, `P`, `F`, `figures`

    Examples::
        >>> from KoBERTScore.tasks import find_best_layer
        >>> from KoBERTScore.utils import train_idf, idf_numpy_to_embed
        >>> from Korpora import Korpora
        >>> from transformers import BertModel, BertTokenizer
        >>> from bokeh.plotting import show, output_notebook

        >>> output_notebook()

        >>> model_name = "beomi/kcbert-base"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> encoder = BertModel.from_pretrained(model_name)

        >>> corpus = Korpora.load('korsts')
        >>> references = corpus.train.texts
        >>> candidates = corpus.train.pairs
        >>> qualities = [float(s) for s in corpus.train.labels]

        >>> idf = train_idf(tokenizer, references)
        >>> idf_embed = idf_numpy_to_embed(idf)

        >>> best_layer, informations = find_best_layer(
        >>>     tokenizer, encoder, references, candidates, qualities,
        >>>     idf=idf_embed, rescale_base=0, model_name='KcBERT + STS', batch_size=256)

        >>> show(informations['figures']['F'])
    """

    def dict_to_array(d):
        return np.array([d[key] for key in sorted(d)])

    R, P, F = correlation(
        bert_tokenizer, bert_model, references, candidates,
        qualities, idf, rescale_base, batch_size)
    R = dict_to_array(R)
    P = dict_to_array(P)
    F = dict_to_array(F)
    best_layer = F.argmax()
    figures = None

    if draw_plot:
        rescale_base = float(rescale_base)
        prefix = '{}'.format(model_name if isinstance(model_name, str) else '')
        figures = {
            'R': lineplot(R, legend=model_name, y_name='R', title=f'{prefix} R, b={rescale_base:.4}'.strip()),
            'P': lineplot(P, legend=model_name, y_name='P', title=f'{prefix} P, b={rescale_base:.4}'.strip()),
            'F': lineplot(F, legend=model_name, y_name='F', title=f'{prefix} F, b={rescale_base:.4}'.strip())
        }
        rpf = gridplot([[
            lineplot(R, legend=model_name, y_name='R', title=f'{prefix} R, b={rescale_base:.4}'.strip()),
            lineplot(P, legend=model_name, y_name='P', title=f'{prefix} P, b={rescale_base:.4}'.strip()),
            lineplot(F, legend=model_name, y_name='F', title=f'{prefix} F, b={rescale_base:.4}'.strip())
        ]])
        figures['RPF'] = rpf

    informations = {'R': R, 'P': P, 'F': F, 'figures': figures}

    return best_layer, informations


def compute_average_l2_norm(bert_tokenizer, bert_model, references, model_name=None, batch_size=128, draw_plot=True):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        references (list of str) : Input sentences
        batch_size (int) : Batch size, default = 128
        draw_plot (Boolean) : If True, it returns bokeh plots

    Returns:
        l2_norm (list of float) : Average l2 norm, length == n_layers + 1
        lineplot (bokeh.figure) : Line plot of `l2_norm`

    Examples::
        >>> from KoBERTScore.tasks import find_best_layer
        >>> from KoBERTScore.utils import train_idf, idf_numpy_to_embed
        >>> from Korpora import Korpora
        >>> from transformers import BertModel, BertTokenizer
        >>> from bokeh.plotting import show, output_notebook

        >>> output_notebook()

        >>> model_name = "beomi/kcbert-base"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> encoder = BertModel.from_pretrained(model_name)

        >>> corpus = Korpora.load('korsts')
        >>> references = corpus.train.texts

        >>> l2_norm, lineplot = average_l2_norm(
        >>>     tokenizer, encoder, references, model_name='KcBERT + STS', batch_size=256)

        >>> show(lineplot)
    """
    n_layers = bert_model.config.num_hidden_layers + 1
    layer_l2norm = [0] * n_layers
    layer_weight = [0] * n_layers

    n_examples = len(references)
    n_batch = math.ceil(n_examples / batch_size)
    for step in tqdm(range(n_batch), desc='Calculating R, P, F', total=n_batch):
        b = step * batch_size
        e = min((step + 1) * batch_size, n_examples)
        refer_batch = references[b: e]
        refer_ids, refer_attention_mask, refer_weight_mask = sents_to_tensor(bert_tokenizer, refer_batch)
        refer_embeds = bert_forwarding(bert_model, refer_ids, refer_attention_mask, output_layer_index='all')
        for layer in range(n_layers):
            l2norm = torch.norm(refer_embeds[layer], p=2, dim=2)
            l2norm = float(((l2norm * refer_weight_mask).sum()).detach())
            weight = float(refer_weight_mask.sum().detach())
            layer_l2norm[layer] += l2norm
            layer_weight[layer] += weight
    layer_l2norm = [norm / weight for norm, weight in zip(layer_l2norm, layer_weight)]


    figures = None
    if draw_plot:
        prefix = '{}'.format(model_name if isinstance(model_name, str) else '')
        figure = lineplot(np.array(layer_l2norm), legend=model_name, y_name='L2 norm', title=f'Average L2-norm')

    return layer_l2norm, figure
