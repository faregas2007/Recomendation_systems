from recsys import eval

def test_get_metrics():
    """
    performance = eval.get_metrics(model, dataloader, top_k, y_true, y_pred)
    assert performance["overall"]["precision"] == 7/10.
    assert performance["overall"]["recall"] == 3/4.
    """
    pass