import torch



def reparameterize(mu, logvar):
    """Samples item from a distribution in a way that allows backpropagation to flow through.

    Args:
        mu: (N, M), Mean of the distribution.
        logvar: (N, M), Log variance of the distribution.

    Returns:
        (N, M), Item sampled from the distribution.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
