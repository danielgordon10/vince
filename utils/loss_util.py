import torch
from dg_util.python_utils import pytorch_util as pt_util

USE_FLOAT = None


def similarity_cross_entropy(similarities, temperature, n_feat, n_rows1, mask=None, n_positives_per_row=None):
    global USE_FLOAT
    similarities = similarities / temperature
    if mask is None:
        assert n_positives_per_row is not None
        # Default identity mask
        mask = (
            torch.eye(n_feat, device=similarities.device, dtype=torch.bool)
                .repeat_interleave(n_positives_per_row, 1)
                .repeat_interleave(n_rows1, 0)
        )

    assert mask.shape == similarities.shape
    similarities = pt_util.split_dim(similarities, 0, n_feat, n_rows1)
    mask = pt_util.split_dim(mask, 0, n_feat, n_rows1)

    # log similarity over (self + all other entries as denom)
    row_maxes = torch.max(similarities, dim=-1, keepdim=True)[0]
    scaled_similarities = similarities - row_maxes

    mask_row_sum = mask.sum(-1)
    if USE_FLOAT is None:
        USE_FLOAT = mask_row_sum.min() != mask_row_sum.max()
    if USE_FLOAT:
        float_mask = mask.float()
        inv_float_mask = 1 - float_mask
        neg_similarities = scaled_similarities * inv_float_mask + -2 ** 20 * float_mask
        pos_similarities = scaled_similarities * float_mask + -2 ** 20 * inv_float_mask
    else:
        # Same number of items per row
        neg_similarities = scaled_similarities[~mask].view(n_feat, n_rows1, -1)
        pos_similarities = scaled_similarities[mask].view(n_feat, n_rows1, mask.shape[2] - neg_similarities.shape[2])

    neg_similarities_exp = torch.exp(neg_similarities).sum(-1, keepdim=True)

    pos_similarities_exp = torch.exp(pos_similarities)
    similarity_log_softmax = pos_similarities - torch.log(pos_similarities_exp + neg_similarities_exp)
    dists = -similarity_log_softmax
    softmax_weights = torch.exp(similarity_log_softmax.detach())

    if USE_FLOAT:
        dists_mean = dists[mask].mean()
        softmax_weight = softmax_weights[mask].mean()
    else:
        dists_mean = dists.mean()
        softmax_weight = softmax_weights.mean()

    return dict(
        # similarity_log_softmax=similarity_log_softmax.mean(),
        dists=dists,
        dist=dists_mean,
        # similarity_raw_scores=similarity_raw_scores.mean(),
        # similarities=similarities.mean(),
        softmax_weights=softmax_weights,
        softmax_weight=softmax_weight,
    )
