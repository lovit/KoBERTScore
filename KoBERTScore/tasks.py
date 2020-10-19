import math
import numpy as np
import torch
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.models import HoverTool, SaveTool
from bokeh.palettes import Blues256
from bokeh.plotting import figure
from bokeh.transform import dodge
from scipy.stats import pearsonr
from tqdm import tqdm

from .score import sents_to_tensor, bert_forwarding
from .score import compute_pairwise_cosine, compute_RPF


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


    figure = None
    if draw_plot:
        prefix = '{}'.format(model_name if isinstance(model_name, str) else '')
        figure = lineplot(np.array(layer_l2norm), legend=model_name, y_name='L2 norm', title=f'Average L2-norm')

    return layer_l2norm, figure


def correlation(bert_tokenizer, bert_model, references, candidates, qualities,
                idf=None, rescale_base=0, batch_size=128):
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

    Returns:
        R (dict) : {layer: correlation}
        P (dict) : {layer: correlation}
        F (dict) : {layer: correlation}

    Examples::
        >>> from transformers import BertModel, BertTokenizer

        >>> model_name = "beomi/kcbert-base"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> encoder = BertModel.from_pretrained(model_name)

        >>> references = [
        >>>     '날씨는 좋고 할일은 많고 어우 연휴 끝났다',
        >>>     '힘을 내볼까? 잘할 수 있어!',
        >>>     '이 문장은 점수가 낮아야만 합니다']
        >>> candidates = [
        >>>     '날씨가 좋다 하지만 할일이 많다 일해라 인간',
        >>>     '힘내라 잘할 수 있다',
        >>>     '테넷봤나요? 역의역의역은역인가요?']
        >>> qualities = [0.85, 0.98, 0.05]

        >>> R, P, F = correlation(tokenizer, encoder, references, candidates, qualities)
        >>> R
        $ {0: 0.9999654999597412,
           1: 0.9992112037241504,
           2: 0.9965136571004495,
           3: 0.9957015840472935,
           4: 0.9988308396315225,
           5: 0.996627590921058,
           6: 0.9945366957299662,
           7: 0.993955314845382,
           8: 0.9934660109682587,
           9: 0.9937264961902929,
           10: 0.9953018679381236,
           11: 0.9985711230470845,
           12: 0.9992405789378926}
    """
    if not isinstance(qualities, np.ndarray):
        qualities = np.array(qualities)

    def corr(array):
        array = np.array(array)
        indices = np.isnan(array)
        if indices.sum() > 0:
            print(f'Found {indices.sum()} NaN values / {array.shape[0]}')
        return pearsonr(qualities[~indices], array[~indices])[0]

    R, P, F = score_from_all_layers(
        bert_tokenizer, bert_model, references, candidates,
        idf, rescale_base, batch_size)

    R = {layer: corr(array) for layer, array in R.items()}
    P = {layer: corr(array) for layer, array in P.items()}
    F = {layer: corr(array) for layer, array in F.items()}
    return R, P, F


def score_from_all_layers(bert_tokenizer, bert_model, references, candidates,
                          idf=None, rescale_base=0, batch_size=128):
    """
    Args:
        bert_tokenizer (transformers.PreTrainedTokenizer)
        bert_model (transformers`s Pretrained models)
        references (list of str) : True sentences
        candidates (list of str) : Generated sentences
        idf (torch.nn.Embedding or None) : IDF weights
        rescale_base (float) : 0 <= rescale_base < 1
            Adjust (R-BERTScore - base) / (1 - base)
        batch_size (int) : Batch size, default = 128

    Returns:
        R (dict) : {layer: list of float}
        P (dict) : {layer: list of float}
        F (dict) : {layer: list of float}

    Examples::
        >>> from transformers import BertModel, BertTokenizer

        >>> model_name = "beomi/kcbert-base"
        >>> tokenizer = BertTokenizer.from_pretrained(model_name)
        >>> encoder = BertModel.from_pretrained(model_name)

        >>> references = [
        >>>     '날씨는 좋고 할일은 많고 어우 연휴 끝났다',
        >>>     '힘을 내볼까? 잘할 수 있어!',
        >>>     '이 문장은 점수가 낮아야만 합니다']
        >>> candidates = [
        >>>     '날씨가 좋다 하지만 할일이 많다 일해라 인간',
        >>>     '힘내라 잘할 수 있다',
        >>>     '테넷봤나요? 역의역의역은역인가요?']

        >>> R, P, F = score_from_all_layers(tokenizer, encoder, references, candidates)
    """

    # Initialize
    n_layers = bert_model.config.num_hidden_layers + 1
    R, P, F = {}, {}, {}
    for layer in range(n_layers):
        R[layer] = []
        P[layer] = []
        F[layer] = []

    n_examples = len(references)
    n_batch = math.ceil(n_examples / batch_size)
    for step in tqdm(range(n_batch), desc='Calculating R, P, F', total=n_batch):
        b = step * batch_size
        e = min((step + 1) * batch_size, n_examples)
        refer_batch = references[b: e]
        candi_batch = candidates[b: e]

        refer_ids, refer_attention_mask, refer_weight_mask = sents_to_tensor(bert_tokenizer, refer_batch)
        candi_ids, candi_attention_mask, candi_weight_mask = sents_to_tensor(bert_tokenizer, candi_batch)

        refer_embeds = bert_forwarding(bert_model, refer_ids, refer_attention_mask, output_layer_index='all')
        candi_embeds = bert_forwarding(bert_model, candi_ids, candi_attention_mask, output_layer_index='all')

        for layer in range(n_layers):
            refer_embeds_i = refer_embeds[layer]
            candi_embeds_i = candi_embeds[layer]
            R_l, P_l, F_l = compute_RPF(
                refer_embeds_i, candi_embeds_i,
                refer_weight_mask, candi_weight_mask,
                refer_ids, candi_ids,
                idf, rescale_base
            )
            R[layer].append(R_l.numpy())
            P[layer].append(P_l.numpy())
            F[layer].append(F_l.numpy())

    R = {layer: np.concatenate(array).tolist() for layer, array in R.items()}
    P = {layer: np.concatenate(array).tolist() for layer, array in P.items()}
    F = {layer: np.concatenate(array).tolist() for layer, array in F.items()}
    return R, P, F


def lineplot(array, legend=None, y_name='', title=None,  color='navy', p=None):
    if p is None:
        tooltips = [('layer', '$x'), (y_name, '$y')]
        tools = [HoverTool(names=["circle"]), SaveTool()]

        p = figure(height=400, width=400, tooltips=tooltips, tools=tools)
        p.xaxis.axis_label = 'Layer'
        p.yaxis.axis_label = y_name

    source = ColumnDataSource(data=dict(
        x = list(range(array.shape[0])),
        y = array
    ))

    if isinstance(legend, str):
        p.vline_stack('y', x='x', color=color, legend_label=legend, source=source)
    else:
        p.vline_stack('y', x='x', color=color, source=source)
    p.circle(x='x', y='y', size=8, color=color, alpha=0.5, name='circle', source=source)
    return p


def plot_bertscore_detail(reference, candidate, bert_tokenizer, bert_model, idf=None,
                          output_layer_index=-1, height='auto', width='auto', title=None, return_gridplot=True):
    """
    Args:
    Examples::
        Loading BERT manually

            >>> from transformers import BertModel, BertTokenizer
            >>> from bokeh.plotting import show
            >>> from KoBERTScore import plot_bertscore_detail

            >>> model_name = "bert-base-uncased"
            >>> tokenizer = BertTokenizer.from_pretrained(model_name)
            >>> encoder = BertModel.from_pretrained(model_name)

            >>> reference = '날씨는 좋고 할일은 많고 어우 연휴 끝났다'
            >>> candidate = '날씨가 좋다 하지만 할일이 많다 일해라 인간'
            >>> p = plot_bertscore_detail(reference, candidate, tokenizer, encoder)
            >>> show(p)

        If env is IPython notebook

            >>> from bokeh.plotting import output_notebook
            >>> output_notebook()
            >>> show(p)

        Using BERTScore class instance

            >>> from KoBERTScore import BERTScore, plot_bertscore_detail
            >>> model_name = "beomi/kcbert-base"
            >>> bertscore = BERTScore(model_name, best_layer=4)
            >>> p = plot_bertscore_detail(
            >>>     reference, candidate, bertscore.tokenizer, bertscore.encoder)
    """
    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError('`reference` and `candidate` must be `str`')

    # tokenization
    refer_ids, refer_attention_mask, refer_weight_mask = sents_to_tensor(bert_tokenizer, [reference])
    candi_ids, candi_attention_mask, candi_weight_mask = sents_to_tensor(bert_tokenizer, [candidate])

    # BERT embedding + Cosine
    refer_embed = bert_forwarding(bert_model, refer_ids, refer_attention_mask, output_layer_index)
    candi_embed = bert_forwarding(bert_model, candi_ids, candi_attention_mask, output_layer_index)
    pairwise_cosine = compute_pairwise_cosine(refer_embed, candi_embed)[0].numpy()

    # set height and width
    if height == 'auto':
        height = max(500, 50 * refer_ids.size()[1])
    if width == 'auto':
        width = max(500, 50 * candi_ids.size()[1] + 50)
    p_cos = draw_pairwise_cosine(bert_tokenizer, refer_ids, candi_ids, pairwise_cosine, title, height, width - 50)
    p_idf = draw_idf(bert_tokenizer, refer_ids, idf, height, width=50)

    if return_gridplot:
        gp = gridplot([[p_cos, p_idf]])
        return gp
    return p_cos, p_idf


def draw_pairwise_cosine(bert_tokenizer, refer_ids, candi_ids, pairwise_cosine, title=None, height=500, width=500):
    refer_vocab = [bert_tokenizer.ids_to_tokens[idx] for idx in refer_ids[0][1: -1].numpy()]
    candi_vocab = [bert_tokenizer.ids_to_tokens[idx] for idx in candi_ids[0][1: -1].numpy()]

    tooltips = [
        ('Reference token', '@refer'),
        ('Candidate token', '@candi'),
        ('Cosine', '@cos')
    ]
    yrange = [f'{i}: {refer_vocab[i]}' for i in range(refer_ids.size()[1] - 2)]
    xrange = [f'{i}: {candi_vocab[i]}' for i in range(candi_ids.size()[1] - 2)]
    p = figure(title=title, height=height, width=width, x_range=xrange, y_range=yrange, tooltips=tooltips)
    p.xaxis.axis_label_text_font_size = '13pt'
    p.yaxis.axis_label_text_font_size = '13pt'

    # Prepare source
    x = []
    y = []
    refers = []
    candis = []
    cos = []
    cos_str = []
    for i_ref, refer in enumerate(refer_ids[0][1: -1].numpy()):
        for i_can, candi in enumerate(candi_ids[0][1: -1].numpy()):
            y.append(f'{i_ref}: {refer_vocab[i_ref]}')
            x.append(f'{i_can}: {candi_vocab[i_can]}')
            refers.append(bert_tokenizer.ids_to_tokens[refer])
            candis.append(bert_tokenizer.ids_to_tokens[candi])
            cos.append(pairwise_cosine[i_ref + 1, i_can + 1])
            cos_str.append(f'{pairwise_cosine[i_ref + 1, i_can + 1]:.3}')
    source = ColumnDataSource(data={
        'x': x, 'y': y, 'refer': refers, 'candi': candis, 'cos': cos, 'cos_str': cos_str
    })

    cmap = LinearColorMapper(palette=list(reversed(Blues256)), low=0.0, high=1.0)
    p.rect('x', 'y', 0.95, 0.95, fill_color={'field': 'cos', 'transform': cmap},
           line_color=None, fill_alpha=0.7, source=source)
    p.text(dodge("x", -0.3, range=p.x_range),
           dodge("y", -0.1, range=p.y_range),
           text='cos_str', text_font_size='13px', source=source)
    return p


def draw_idf(bert_tokenizer, refer_ids, idf, height=500, width=50):
    if idf is None:
        n_vocab = len(bert_tokenizer)
        weight = torch.ones((n_vocab, 1), dtype=torch.float)
        idf = torch.nn.Embedding(n_vocab, 1, _weight=weight)

    tooltips = [
        ('Reference token', '@refer'),
        ('IDF', '@idf')
    ]
    refer_vocab = [bert_tokenizer.ids_to_tokens[idx] for idx in refer_ids[0][1: -1].numpy()]
    yrange = [f'{i}: {refer_vocab[i]}' for i in range(refer_ids.size()[1] - 2)]
    xrange = ['IDF']
    p = figure(height=height, width=width, x_range=xrange, y_range=yrange, tooltips=tooltips, tools=[])
    p.yaxis.visible = False

    y = []
    refers = []
    idf = list(idf(refer_ids[0][1: -1]).detach().numpy().reshape(-1))
    idf_str = []
    for i_ref, refer in enumerate(refer_ids[0][1: -1].numpy()):
        y.append(f'{i_ref}: {refer_vocab[i_ref]}')
        refers.append(bert_tokenizer.ids_to_tokens[refer])
        idf_str.append(f'{idf[i_ref]:.3}')
    x = ['IDF'] * len(y)
    source = ColumnDataSource(data={
        'x': x, 'y': y, 'refer': refers, 'idf': idf, 'idf_str': idf_str
    })

    p.rect('x', 'y', 0.95, 0.95, line_color=None, fill_color='white', source=source)
    p.text(dodge("x", -0.3, range=p.x_range),
           dodge("y", -0.1, range=p.y_range),
           text='idf_str', text_font_size='13px', source=source)
    return p
