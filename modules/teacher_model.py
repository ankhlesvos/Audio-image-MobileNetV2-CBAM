# modules/teacher_model.py
#
# Audio Spectrogram Transformer (AST) wrapper for teacher model in knowledge distillation.
#
# Design decisions:
#   - Teacher uses single-channel Log-Mel (channel 0 of the student's 3-ch input).
#     The teacher's role is producing strong class distributions; input homogeneity
#     with the student is not required.
#   - Built on HuggingFace ASTForAudioClassification for minimal adaptation code.
#   - The AST config is adjusted to match our spectrogram dimensions (160 mel × 157 time)
#     and position embeddings are interpolated from the pretrained checkpoint.
#   - Exposes get_features() to return [CLS] token embedding (768-d) for optional
#     feature distillation.

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ASTForAudioClassification, ASTConfig


class ASTTeacherModel(nn.Module):
    """
    AST-based teacher model for ship sound classification.

    Wraps HuggingFace ASTForAudioClassification with:
      - Automatic input adaptation: takes [B, 3, H, W] from the existing dataloader,
        extracts channel 0 (Log-Mel), and reshapes for AST.
      - Custom classification head for the target number of classes.
      - Position embedding interpolation to handle non-standard input dimensions.
      - Feature access for intermediate-layer distillation.

    Args:
        num_classes: Number of output classes (default: 4 for DeepShip: Cargo, Passengership, Tanker, Tug).
        pretrained_name: HuggingFace model name or path for pretrained AST weights.
        freeze_encoder: If True, freeze the AST encoder (only train classifier head).
        num_mel_bins: Number of mel frequency bins in the input (default: 160).
        max_length: Number of time frames in the input (default: 157).
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_encoder: bool = False,
        num_mel_bins: int = 160,
        max_length: int = 157,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_name = pretrained_name
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length

        # Step 1: Load pretrained config and weights with ORIGINAL dimensions
        # (we need the original position embeddings for interpolation)
        orig_config = ASTConfig.from_pretrained(pretrained_name)

        # Step 2: Create a new config with OUR dimensions + num_classes
        config = ASTConfig.from_pretrained(pretrained_name)
        config.num_labels = num_classes
        config.num_mel_bins = num_mel_bins
        config.max_length = max_length

        # Build model with new config (position embeddings will be initialized for our dims)
        self.ast = ASTForAudioClassification(config)

        # Step 3: Load pretrained weights, interpolating position embeddings
        self._load_pretrained_with_interpolation(pretrained_name, orig_config, config)

        # The hidden size of the AST encoder (typically 768)
        self.hidden_size = config.hidden_size

        if freeze_encoder:
            self._freeze_encoder()

        # Cache for the last [CLS] embedding (populated during forward)
        self._last_cls_embedding: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] where C >= 1.
               Channel 0 is assumed to be the Log-Mel spectrogram.

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        # Extract single-channel Log-Mel → [B, H, W]
        mel = x[:, 0, :, :]  # [B, H, W]

        # AST expects input_values of shape [B, max_length, num_mel_bins].
        # Our mel is [B, n_mels, time_frames] → transpose to [B, time_frames, n_mels]
        mel = mel.transpose(1, 2)  # [B, T, n_mels]

        # Run encoder to get hidden states and pooler output
        encoder = self.ast.audio_spectrogram_transformer
        enc_output = encoder(input_values=mel)

        # Cache [CLS] token embedding for feature distillation
        # pooler_output is the [CLS] after dense+tanh in the AST pooler
        self._last_cls_embedding = enc_output.pooler_output

        # Run the classifier head on pooler_output
        logits = self.ast.classifier(enc_output.pooler_output)

        return logits

    def get_features(self) -> torch.Tensor:
        """
        Returns the [CLS] token embedding from the last forward pass.

        Shape: [B, hidden_size]  (typically [B, 768])
        Must be called AFTER a forward() call.
        """
        if self._last_cls_embedding is None:
            raise RuntimeError(
                "get_features() called before forward(). "
                "Run a forward pass first."
            )
        return self._last_cls_embedding

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pretrained_with_interpolation(
        self,
        pretrained_name: str,
        orig_config: ASTConfig,
        new_config: ASTConfig,
    ):
        """
        Load pretrained weights with position embedding interpolation.

        The AST pretrained model has position embeddings sized for its original
        input dimensions (e.g., 128 mel × 1024 time). We interpolate them to
        match our custom dimensions (e.g., 160 mel × 157 time).
        """
        from transformers import ASTForAudioClassification as _AST

        # Load the full pretrained model with original config
        pretrained = _AST.from_pretrained(pretrained_name)
        pretrained_state = pretrained.state_dict()
        model_state = self.ast.state_dict()

        loaded, skipped = 0, 0
        for name, param in pretrained_state.items():
            if name not in model_state:
                skipped += 1
                continue

            if model_state[name].shape == param.shape:
                model_state[name] = param
                loaded += 1
            elif 'position_embeddings' in name:
                # Interpolate position embeddings
                print(f"  [AST] Interpolating {name}: {param.shape} → {model_state[name].shape}")
                interpolated = self._interpolate_pos_embed(
                    param, model_state[name].shape[1]
                )
                model_state[name] = interpolated
                loaded += 1
            elif 'classifier' in name:
                # Skip classifier — we reinitialize it for our num_classes
                skipped += 1
            else:
                print(f"  [AST] Skipping {name}: shape mismatch {param.shape} vs {model_state[name].shape}")
                skipped += 1

        self.ast.load_state_dict(model_state, strict=False)
        print(f"  [AST] Loaded {loaded} params, skipped {skipped}")

        # Clean up
        del pretrained
        del pretrained_state

    @staticmethod
    def _interpolate_pos_embed(
        pos_embed: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Interpolate position embeddings from original length to target length.

        The first 2 tokens are [CLS] and [DIST] tokens — keep them as-is,
        interpolate the rest (patch embeddings).

        Args:
            pos_embed: [1, orig_length, hidden_size]
            target_length: desired total length (including CLS/DIST tokens)

        Returns:
            [1, target_length, hidden_size]
        """
        # pos_embed shape: [1, N, D]
        num_special = 2  # CLS + DIST tokens
        cls_dist = pos_embed[:, :num_special, :]  # [1, 2, D]
        patch_embed = pos_embed[:, num_special:, :]  # [1, N-2, D]

        target_patches = target_length - num_special

        if patch_embed.shape[1] == target_patches:
            return pos_embed

        # Interpolate: treat as 1D signal [1, D, N-2] → [1, D, target_patches]
        patch_embed = patch_embed.transpose(1, 2)  # [1, D, N-2]
        patch_embed = F.interpolate(
            patch_embed, size=target_patches, mode='linear', align_corners=False
        )
        patch_embed = patch_embed.transpose(1, 2)  # [1, target_patches, D]

        return torch.cat([cls_dist, patch_embed], dim=1)

    def _freeze_encoder(self):
        """Freeze all encoder parameters; only the classifier head is trainable."""
        for name, param in self.ast.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all parameters (call after initial frozen epochs)."""
        for param in self.ast.parameters():
            param.requires_grad = True


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Instantiating ASTTeacherModel (num_classes=3, mel=160, time=157)...")
    model = ASTTeacherModel(num_classes=3, num_mel_bins=160, max_length=157)
    print(f"Model hidden_size: {model.hidden_size}")

    # Simulate input from AudioDataset: [B, 3, n_mels, time_frames]
    dummy = torch.randn(2, 3, 160, 157)
    logits = model(dummy)
    print(f"Logits shape: {logits.shape}")  # expect [2, 3]

    feats = model.get_features()
    print(f"Feature shape: {feats.shape}")  # expect [2, 768]

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total / 1e6:.2f}M, Trainable: {trainable / 1e6:.2f}M")

    print("Teacher model test passed ✓")
