import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _diagonal_targets(x: Tensor, y: Tensor) -> Tensor:
    """In-batch CE targets for x[i] matched to y[i * target_per_x + ...]."""
    if y.size(0) % x.size(0) != 0:
        raise ValueError(
            f"Target count {y.size(0)} must be a multiple of source count {x.size(0)}"
        )
    target_per_x = y.size(0) // x.size(0)
    return torch.arange(
        0,
        x.size(0) * target_per_x,
        target_per_x,
        device=x.device,
        dtype=torch.long,
    )


class SimpleContrastiveLoss:
    """CLIP-style contrastive loss with optional bi-directional averaging."""

    def __init__(self, temperature: float = 0.02, bidirectional: bool = True):
        self.temperature = temperature
        self.bidirectional = bidirectional
        self.losses: dict[str, float] = {}

    @staticmethod
    def _round_losses(components: dict[str, Tensor]) -> dict[str, float]:
        return {k: round(v.detach().float().item(), 4) for k, v in components.items()}

    def _directional_loss(
        self,
        x: Tensor,
        y: Tensor,
        target: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        if target is None:
            target = _diagonal_targets(x, y)
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits / self.temperature, target, reduction=reduction)

    def loss_components(
        self,
        x: Tensor,
        y: Tensor,
        target: Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        loss_q2t = self._directional_loss(x, y, target=target, reduction=reduction)
        components = {"loss_q2t": loss_q2t}
        if self.bidirectional:
            loss_t2q = self._directional_loss(y, x, reduction=reduction)
            components["loss_t2q"] = loss_t2q
            components["loss"] = 0.5 * (loss_q2t + loss_t2q)
        else:
            components["loss"] = loss_q2t
        return components

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        target: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        components = self.loss_components(x, y, target=target, reduction=reduction)
        self.losses = self._round_losses(components)
        return components["loss"]


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(
        self,
        n_target: int = 0,
        scale_loss: bool = True,
        temperature: float = 0.02,
        bidirectional: bool = True,
    ):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature, bidirectional=bidirectional)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        components = self.loss_components(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            components = {k: v * self.word_size for k, v in components.items()}
        self.losses = self._round_losses(components)
        return components["loss"]

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


def build_contrastive_loss(
    *,
    is_ddp: bool,
    temperature: float,
    bidirectional: bool,
) -> SimpleContrastiveLoss:
    """Factory used by CLIP and CCLIP trainers."""
    cls = DistributedContrastiveLoss if is_ddp else SimpleContrastiveLoss
    return cls(temperature=temperature, bidirectional=bidirectional)
