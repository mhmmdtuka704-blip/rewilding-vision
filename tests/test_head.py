import torch
from rewilding_vision.segformer import make_residual_head


def test_zero_residual_head_preserves_pretrained_vegetation_probability():
    torch.manual_seed(16)
    classifier = torch.nn.Conv2d(4, 150, 1)
    features = torch.randn(2, 4, 10, 12)
    vegetation = [4, 9, 17, 66, 72]
    adapted = make_residual_head(classifier, vegetation)
    with torch.no_grad():
        expected = classifier(features).softmax(1)[:, vegetation].sum(1)
        actual = adapted(features).softmax(1)[:, 1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert sum(p.numel() for p in adapted.residual.parameters()) == 10
