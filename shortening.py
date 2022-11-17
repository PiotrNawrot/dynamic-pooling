import torch


def final(foo,
          upsample):
    """
        Input:
            B x L x S
    """
    autoregressive = foo != 0
    lel = 1 - foo

    lel[autoregressive] = 0

    dim = 2 if upsample else 1

    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

    return lel


def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()
    assert n_segments > 0

    if upsample:
        n_segments += 1

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )

    hh1 = boundaries.cumsum(1)

    if not upsample:
        hh1 -= boundaries

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample(boundaries, hidden, null_group):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - We append an extra "null" group at the beginning
        - We discard last group because it won't be used (in terms of upsampling)

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D
    """

    foo = common(boundaries, upsample=False)  # B x L x S
    bar = final(foo=foo, upsample=False)  # B x L x S

    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
    shortened_hidden = torch.cat(
        [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
    )

    return shortened_hidden


def upsample(boundaries, shortened_hidden):
    """
        Upsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - i-th group can be upsampled only to the tokens from (i+1)-th group, otherwise there's a leak

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D
    """

    foo = common(boundaries, upsample=True)  # B x L x S
    bar = final(foo, upsample=True)  # B x L x S

    return torch.einsum('sbd,bls->lbd', shortened_hidden, bar)
