"""ConceptNegation: one-time closed-form T_unk decontamination at T2 init.

During T1, novel objects were lumped into the unknown class, so T_unk may
contain spurious components along novel-class directions. ConceptNegation
removes these via orthogonal projection subtraction:

    t_unk_refined = t_unk - alpha * V^T @ (V @ t_unk)

where V = L2-normalized novel-class embeddings (N x D).
"""

import torch
import torch.nn.functional as F


class ConceptNegation:
    """One-time T_unk decontamination via novel-subspace projection removal.

    Args:
        alpha (float): Projection removal strength. 1.0 = full removal.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def refine(self, t_unk: torch.Tensor, novel_embeddings: torch.Tensor) -> torch.Tensor:
        """Remove novel-subspace components from T_unk in-place.

        Args:
            t_unk: (D,) unknown class embedding.
            novel_embeddings: (N, D) novel class embeddings.

        Returns:
            Refined T_unk embedding (D,).
        """
        V = F.normalize(novel_embeddings.detach(), dim=-1)  # (N, D)
        proj = V.T @ (V @ t_unk.detach())  # (D,)
        refined = t_unk.data - self.alpha * proj.to(t_unk.device)

        old_norm = t_unk.data.norm().item()
        overlap = float(F.cosine_similarity(t_unk.data.unsqueeze(0),
                                            proj.unsqueeze(0)).item())
        t_unk.data.copy_(refined)
        new_norm = t_unk.data.norm().item()

        print(f"[ConceptNegation] alpha={self.alpha} "
              f"novel_overlap={overlap:.4f} "
              f"norm: {old_norm:.4f} -> {new_norm:.4f}")
        return t_unk
