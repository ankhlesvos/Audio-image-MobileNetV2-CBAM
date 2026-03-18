# modules/distillation.py
#
# Knowledge distillation loss functions.
#
# Two-level distillation:
#   1. Response Distillation (output-layer): KL divergence between softened logits.
#      L = (1 - α) * L_hard + α * T² * KL(student_soft || teacher_soft)
#
#   2. Feature Distillation (intermediate-layer): MSE between projected student
#      and teacher features.
#      L_feature = MSE(proj(student_feat), teacher_feat)
#
# Combined: L_total = (1 - α) * L_hard + α * T² * L_KD + β * L_feature

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseDistillationLoss(nn.Module):
    """
    KL-divergence-based response distillation loss.

    Computes KL(softened_student || softened_teacher) at a given temperature.
    Note: this returns ONLY the KD component. The caller handles combining
    with the hard loss using alpha.

    Args:
        temperature: Softmax temperature for softening logits (default: 4.0).
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, C] raw logits from student model.
            teacher_logits: [B, C] raw logits from teacher model.

        Returns:
            Scalar KD loss (already scaled by T²).
        """
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence * T² (standard KD scaling)
        kd_loss = self.kl_div(student_soft, teacher_soft) * (T * T)
        return kd_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss with a learnable projection head.

    Maps student features to teacher feature dimension, then computes
    MSE (or cosine) loss.

    Args:
        student_dim: Dimensionality of student feature (e.g., 1280 for MobileNetV2).
        teacher_dim: Dimensionality of teacher feature (e.g., 768 for AST).
        loss_type: 'mse' or 'cosine' (default: 'mse').
    """

    def __init__(
        self,
        student_dim: int = 1280,
        teacher_dim: int = 768,
        loss_type: str = 'mse',
    ):
        super().__init__()
        self.loss_type = loss_type

        # Projection head: student_dim → teacher_dim
        self.projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim, teacher_dim),
        )

        if loss_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_features: [B, student_dim] pre-classifier features.
            teacher_features: [B, teacher_dim] teacher embeddings.

        Returns:
            Scalar feature distillation loss.
        """
        projected = self.projector(student_features)

        if self.loss_type == 'cosine':
            # target = +1 (maximize similarity)
            target = torch.ones(projected.size(0), device=projected.device)
            return self.criterion(projected, teacher_features.detach(), target)
        else:
            return self.criterion(projected, teacher_features.detach())


class CombinedKDLoss(nn.Module):
    """
    Combined Knowledge Distillation loss.

    L_total = (1 - α) * L_hard + α * L_KD_response + β * L_feature

    Args:
        temperature: Softmax temperature for response distillation.
        alpha: Weight for response distillation (0.0 = hard only, 1.0 = KD only).
        beta: Weight for feature distillation (0.0 = disabled).
        student_dim: Student feature dimension (for feature distillation).
        teacher_dim: Teacher feature dimension (for feature distillation).
        feature_loss_type: 'mse' or 'cosine'.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.0,
        student_dim: int = 1280,
        teacher_dim: int = 768,
        feature_loss_type: str = 'mse',
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.response_kd = ResponseDistillationLoss(temperature=temperature)

        if beta > 0:
            self.feature_kd = FeatureDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                loss_type=feature_loss_type,
            )
        else:
            self.feature_kd = None

    def forward(
        self,
        hard_loss: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hard_loss: Scalar loss from standard CE/Focal with true labels.
            student_logits: [B, C] student raw logits.
            teacher_logits: [B, C] teacher raw logits (detached).
            student_features: [B, student_dim] (optional, for feature KD).
            teacher_features: [B, teacher_dim] (optional, for feature KD).

        Returns:
            Combined scalar loss.
        """
        # Response distillation
        kd_loss = self.response_kd(student_logits, teacher_logits)

        total = (1 - self.alpha) * hard_loss + self.alpha * kd_loss

        # Feature distillation (optional)
        if self.feature_kd is not None and student_features is not None and teacher_features is not None:
            feat_loss = self.feature_kd(student_features, teacher_features)
            total = total + self.beta * feat_loss

        return total


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing distillation losses...")
    B, C = 4, 3

    # Response distillation
    rd = ResponseDistillationLoss(temperature=4.0)
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    loss_rd = rd(s_logits, t_logits)
    print(f"Response KD loss: {loss_rd.item():.4f}")

    # Feature distillation
    fd = FeatureDistillationLoss(student_dim=1280, teacher_dim=768, loss_type='mse')
    s_feat = torch.randn(B, 1280)
    t_feat = torch.randn(B, 768)
    loss_fd = fd(s_feat, t_feat)
    print(f"Feature KD loss (MSE): {loss_fd.item():.4f}")

    # Cosine feature distillation
    fd_cos = FeatureDistillationLoss(student_dim=1280, teacher_dim=768, loss_type='cosine')
    loss_cos = fd_cos(s_feat, t_feat)
    print(f"Feature KD loss (Cosine): {loss_cos.item():.4f}")

    # Combined loss
    combined = CombinedKDLoss(
        temperature=4.0, alpha=0.5, beta=1.0,
        student_dim=1280, teacher_dim=768
    )
    hard_loss = torch.tensor(1.5)
    total = combined(hard_loss, s_logits, t_logits, s_feat, t_feat)
    print(f"Combined KD loss: {total.item():.4f}")

    # Combined without feature distillation
    combined_no_feat = CombinedKDLoss(temperature=4.0, alpha=0.5, beta=0.0)
    total_no_feat = combined_no_feat(hard_loss, s_logits, t_logits)
    print(f"Combined KD loss (no feature): {total_no_feat.item():.4f}")

    print("All distillation tests passed ✓")
