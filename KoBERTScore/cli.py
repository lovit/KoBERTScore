import argparse
import os
import numpy as np
import torch
import warnings
from bokeh.io import save
from Korpora import Korpora
from transformers import BertModel, BertTokenizer

from .about import __name__, __version__
from .score import train_idf, idf_numpy_to_embed
from .tasks import find_best_layer, compute_average_l2_norm, score_from_all_layers


def main():
    parser = argparse.ArgumentParser(description='Ko-BERTScore Command Line Interface')
    subparsers = parser.add_subparsers(help='Ko-BERTScore tasks')

    # Show package version
    parser_version = subparsers.add_parser('version', help='Show package version')
    parser_version.set_defaults(func=version)

    # Finding best layer
    parser_best_layer = subparsers.add_parser('best_layer', help='Finding best layer')
    parser_best_layer.add_argument('--model_name_or_path', type=str, required=True, help='BERT model path or name')
    parser_best_layer.add_argument('--device', type=str, default=None, help='cpu, cuda, cuda:0, or None')
    # Will be implemented
#     parser_best_layer.add_argument('--references', type=str, help='References path')
#     parser_best_layer.add_argument('--candidates', type=str, help='Candidates path')
#     parser_best_layer.add_argument('--qualities', type=str, help='Qualities path')
#     parser_best_layer.add_argument('--idf_path', type=str, default=None, help='Pretrained IDF path')
    parser_best_layer.add_argument('--corpus', type=str, choices=['korsts'], help='STS corpus in Korpora')
    parser_best_layer.add_argument('--rescale_base', type=float, default=0.0, help='Rescale base value')
    parser_best_layer.add_argument('--batch_size', type=int, default=128, help='BERT embedding batch size')
    parser_best_layer.add_argument('--draw_plot', dest='draw_plot', action='store_true')
    parser_best_layer.add_argument('--output_dir', type=str, default=None, help='Directory for saving figures')
    parser_best_layer.set_defaults(func=best_layer)

    # Find rescale base
    parser_rescale_base = subparsers.add_parser('rescale_base', help='Finding rescale base value')
    parser_rescale_base.add_argument('--model_name_or_path', type=str, required=True, help='BERT model path or name')
    parser_rescale_base.add_argument('--device', type=str, default=None, help='cpu, cuda, cuda:0, or None')
    parser_rescale_base.add_argument('--references', type=str, help='References path')
    parser_rescale_base.add_argument('--output_path', type=str, default=None, help='Result file path')
    parser_rescale_base.add_argument('--batch_size', type=int, default=128, help='BERT embedding batch size')
    # Will be implemented
#     parser_rescale_base.add_argument('--idf_path', type=str, default=None, help='Pretrained IDF path')
    parser_rescale_base.set_defaults(func=rescale_base)

    # Compute average l2 norm of every BERT layer out
    parser_l2norm = subparsers.add_parser('l2norm', help='Compute average l2 norm of every BERT layer out')
    parser_l2norm.add_argument('--model_name_or_path', type=str, required=True, help='BERT model path or name')
    parser_l2norm.add_argument('--device', type=str, default=None, help='cpu, cuda, cuda:0, or None')
    parser_l2norm.add_argument('--references', type=str, help='References path')
    parser_l2norm.add_argument('--output_path', type=str, default=None, help='Result file path')
    parser_l2norm.add_argument('--batch_size', type=int, default=128, help='BERT embedding batch size')
    parser_l2norm.add_argument('--draw_plot', dest='draw_plot', action='store_true')
    parser_l2norm.set_defaults(func=average_l2_norm)

    args = parser.parse_args()
    task_function = args.func
    task_function(args)


def version(args):
    print(f'{__name__}=={__version__}')


def best_layer(args):
    print(f'Finding best performance BERT layer with {args.model_name_or_path}')

    if not -1 < args.rescale_base < 1:
        raise ValueError("`rescale_base` must be in [-1, 1]")
    if args.draw_plot and args.output_dir is None:
        raise ValueError('Set `output_dir` when use `draw_plot`')

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    encoder = BertModel.from_pretrained(args.model_name_or_path).to(device)

    # Load STS corpus
    corpus = Korpora.load('korsts')
    references = []
    candidates = []
    qualities = []
    for data in (corpus.train, corpus.dev):
        for example in data:
            references.append(example.text)
            candidates.append(example.pair)
            qualities.append(example.label)

    # Train IDF
    idf = train_idf(tokenizer, references)
    idf_embed = idf_numpy_to_embed(idf)

    # Find best layer
    model_name = args.model_name_or_path.split(os.path.sep)[-1]
    best_layer, informations = find_best_layer(
        tokenizer, encoder, references, candidates, qualities,
        idf=idf_embed, rescale_base=args.rescale_base,
        model_name=model_name, batch_size=args.batch_size
    )

    print(f'  - Best performance layer : {best_layer}')

    # Save figures
    if args.draw_plot:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.getcwd()
        dirname = os.path.abspath(output_dir)
        print(f'Saving figures at {dirname}')
        os.makedirs(dirname, exist_ok=True)
        warnings.filterwarnings("ignore")
        save(informations['figures']['R'], f'{dirname}/R.html')
        save(informations['figures']['P'], f'{dirname}/P.html')
        save(informations['figures']['F'], f'{dirname}/F.html')
        save(informations['figures']['RPF'], f'{dirname}/RPF.html')


def rescale_base(args):
    print(f'Finding rescale base BERT layer with {args.model_name_or_path}')

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    encoder = BertModel.from_pretrained(args.model_name_or_path).to(device)

    # Load references
    with open(args.references, encoding='utf-8') as f:
        references = [line.strip() for line in f]

    # Generate candidates from permutating references
    n_sents = len(references)
    candidates = [references[idx] for idx in np.random.permutation(n_sents)]

    # Train IDF
    idf = train_idf(tokenizer, references)
    idf_embed = idf_numpy_to_embed(idf)

    # Average scores
    def average(scores):
        scores = np.array(scores)
        indices = ~np.isnan(scores)
        return scores[indices].mean()

    R, P, F = score_from_all_layers(
        tokenizer, encoder, references, candidates,
        idf_embed, rescale_base=0, batch_size=args.batch_size
    )
    R = {layer: average(scores) for layer, scores in R.items()}
    P = {layer: average(scores) for layer, scores in P.items()}
    F = {layer: average(scores) for layer, scores in F.items()}

    # Reporting
    n_layers = len(R)
    form = '| {} | {} | {} | {} |'
    report = [form.format('layer', 'R', 'P', 'F'), form.format('---', '---', '---', '---')]
    for layer in range(n_layers):
        report.append(form.format('{:2}'.format(layer), R[layer], P[layer], F[layer]))
    report = '\n'.join(report)
    print(report)

    # Write report
    if args.output_path is not None:
        dirname = os.path.abspath(os.path.dirname(args.output_path))
        print(f'Saving rescale base at {dirname}')
        os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write(report)


def average_l2_norm(args):
    print(f'Compute average L2 norm in every layer of BERT {args.model_name_or_path}')

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    encoder = BertModel.from_pretrained(args.model_name_or_path).to(device)

    # Load references
    with open(args.references, encoding='utf-8') as f:
        references = [line.strip() for line in f]

    layer_l2norm, figure = compute_average_l2_norm(
        tokenizer, encoder, references,
        model_name=args.model_name_or_path.split('/')[-1],
        batch_size=args.batch_size, draw_plot=args.draw_plot)

    # Reporting
    form = '| {} | {} |'
    report = [form.format('layer', 'L2 norm'), form.format('---', '---')]
    for layer, l2norm in enumerate(layer_l2norm):
        report.append(form.format('{:2}'.format(layer), l2norm))
    report = '\n'.join(report)
    print(report)

    # Write report
    if args.output_path is not None:
        dirname = os.path.abspath(os.path.dirname(args.output_path))
        print(f'Saving L2 norm at {dirname}')
        os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    if args.draw_plot:
        dirname = os.path.abspath(args.output_path)
        print(f'Saving figure at {dirname}')
        warnings.filterwarnings("ignore")
        save(figure, f'{args.output_path}.html')


if __name__ == '__main__':
    main()
