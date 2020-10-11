import torch
from KoBERTScore.score import compute_pairwise_cosine


def test_pairwise_cosine():
    torch.manual_seed(0)
    input1 = torch.randn(3, 4, 5)
    input2 = torch.randn(3, 7, 5)
    assert list(compute_pairwise_cosine(input1, input2).size()) == [3, 4, 7]

