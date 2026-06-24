import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


def mdn_nll(pi_logits, mu, sigma, target, ent_bonus=0.0, target_jitter=0.0):
    """NLL of target under Gaussian mixture, averaged over batch.
    Uses logsumexp for stability. Optional entropy bonus to stop components
    from collapsing. Target jitter adds noise as regularization.
    """
    log_pi = F.log_softmax(pi_logits, dim=-1)
    t = target.unsqueeze(1)
    if target_jitter > 0:
        t = t + torch.randn_like(t) * target_jitter
    # log probability of target under each Gaussian component
    z = (t - mu) / sigma
    log_comp = -0.5 * (z * z + LOG_2PI) - torch.log(sigma)
    # logsumexp combines log(pi_k) + log(N_k) across components
    log_p = torch.logsumexp(log_pi + log_comp, dim=-1)
    nll = -log_p.mean()
    if ent_bonus > 0:
        # subtract entropy penalty to encourage using all K components
        pi = torch.exp(log_pi)
        ent = -(pi * log_pi).sum(dim=-1).mean()
        nll = nll - ent_bonus * ent
    return nll


def mdn_log_prob(pi_logits, mu, sigma, target):
    """Per-sample log probability under the mixture.
    Same math as mdn_nll but returns per-sample values, not the mean.
    """
    log_pi = F.log_softmax(pi_logits, dim=-1)
    t = target.unsqueeze(1)
    z = (t - mu) / sigma
    log_comp = -0.5 * (z * z + LOG_2PI) - torch.log(sigma)
    return torch.logsumexp(log_pi + log_comp, dim=-1)


def mdn_cdf(pi_logits, mu, sigma, x):
    """CDF of the Gaussian mixture at x. x: (B,) or (B, T).
    sum_k pi_k * Phi((x - mu_k) / sigma_k), using erf for the normal CDF.
    """
    pi = F.softmax(pi_logits, dim=-1)
    if x.dim() == 1:
        z = (x.unsqueeze(1) - mu) / (sigma * math.sqrt(2.0))
        comp = 0.5 * (1.0 + torch.erf(z))
        return (pi * comp).sum(dim=-1)
    else:
        # x: (B, T) - evaluate CDF at multiple points per sample
        z = (x.unsqueeze(2) - mu.unsqueeze(1)) / (sigma.unsqueeze(1) * math.sqrt(2.0))
        comp = 0.5 * (1.0 + torch.erf(z))
        return (pi.unsqueeze(1) * comp).sum(dim=-1)


def mdn_pdf(pi_logits, mu, sigma, x):
    """PDF of the Gaussian mixture at x. x: (B,) or (B, T).
    Used by the Newton bid optimizer to compute the derivative of expected profit.
    """
    pi = F.softmax(pi_logits, dim=-1)
    if x.dim() == 1:
        xz = (x.unsqueeze(1) - mu) / sigma
        comp = torch.exp(-0.5 * xz * xz) / (sigma * math.sqrt(2.0 * math.pi))
        return (pi * comp).sum(dim=-1)
    else:
        xz = (x.unsqueeze(2) - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        comp = torch.exp(-0.5 * xz * xz) / (sigma.unsqueeze(1) * math.sqrt(2.0 * math.pi))
        return (pi.unsqueeze(1) * comp).sum(dim=-1)


def discrete_bins_nll(logits, target_bin):
    """Cross entropy loss on integer bin index.
    The target is the actual payprice clamped to [0, num_bins-1].
    Standard classification loss treating each price level as a class.
    """
    return F.cross_entropy(logits, target_bin)


def discrete_bins_smoothed_nll(logits, target_bin, sigma=1.0):
    """Cross entropy with Gaussian-smoothed targets.
    Spreads probability to neighboring bins so predicting 80 when
    the true price was 81 isn't penalized as hard.
    """
    num_bins = logits.size(1)
    t = target_bin.float().unsqueeze(1)
    centers = torch.arange(num_bins, device=logits.device, dtype=torch.float32).unsqueeze(0)
    # Gaussian weights centered on the true bin
    weights = torch.exp(-0.5 * ((centers - t) / sigma) ** 2)
    weights = weights / weights.sum(dim=1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(weights * log_probs).sum(dim=1).mean()
