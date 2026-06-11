"""Poincaré ball operations for hyperbolic embeddings."""

import torch
from torch import Tensor


def expmap0(u: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """
    Exponential map at the origin of the Poincaré ball.

    Maps unconstrained tangent vectors to points with ||x|| < 1/sqrt(c).
    """
    sqrt_c = c**0.5
    u_norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    coef = (sqrt_c * u_norm).tanh() / (sqrt_c * u_norm)
    return coef * u


def project_to_ball(x: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Project points back into the open Poincaré ball if numerical drift occurs."""
    max_norm = (1.0 - eps) / (c**0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(norm > max_norm, x * (max_norm / norm), x)


def embed_sphere_to_ball(
    x: Tensor,
    *,
    scale: float = 1.0,
    c: float = 1.0,
    eps: float = 1e-5,
) -> Tensor:
    """
    Embed unit-sphere directions into the Poincaré ball via expmap0(scale * x).

    Cached old CLIP features are L2-normalized on the sphere; treating them as
    tangent vectors at the origin places them on a shell inside the ball.
    """
    return project_to_ball(expmap0(scale * x, c=c, eps=eps), c=c, eps=eps)


def poincare_distance(x: Tensor, y: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """
    Pairwise Poincaré distance d_c(x, y) for x [N, d] and y [M, d].

    Returns a [N, M] distance matrix.
    """
    x = project_to_ball(x, c=c, eps=eps)
    y = project_to_ball(y, c=c, eps=eps)

    x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm_sq = y.pow(2).sum(dim=-1, keepdim=True)
    diff_sq = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)

    denom = (1.0 - c * x_norm_sq).clamp_min(eps) * (1.0 - c * y_norm_sq.T).clamp_min(eps)
    xi = 1.0 + 2.0 * c * diff_sq / denom
    return torch.arccosh(xi.clamp_min(1.0 + eps)) / (c**0.5)
