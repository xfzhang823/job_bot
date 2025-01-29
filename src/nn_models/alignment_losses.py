"""
alignment_losses.py
Author: Xiao-Fei Zhang
Last upated: 2025 Jan

Losses Module for the TransformerModel in transformer_model.py
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Device setup for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlignmentLoss(nn.Module):
    """
    A flexible loss function for resume-job alignment tasks, integrating the TransformerModel
    for embedding refinement.

    Attributes:
        - align_loss (nn.CosineEmbeddingLoss): Measures alignment using cosine similarity.
        - substance_loss (nn.MSELoss or None): Ensures refined embeddings preserve original
        structure.
    """

    def __init__(self, include_substance_loss: bool = False) -> None:
        """
        Initialize the loss function with alignment and optional substance components.

        Args:
            include_substance_loss (bool): Whether to include substance loss.
            Default is False.
        """
        super().__init__()
        self.align_loss = (
            nn.CosineEmbeddingLoss()
        )  # Measures alignment with cosine similarity
        self.include_substance_loss = include_substance_loss
        if self.include_substance_loss:
            self.substance_loss = nn.MSELoss()  # Initialize only if enabled
        else:
            self.substance_loss = None  # Make explicit for clarity

    def forward(
        self,
        transformer_model: nn.Module,
        resume_embs: torch.Tensor,
        job_desc_embs: torch.Tensor,
        positions: torch.Tensor,
        entailment_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the hybrid loss using the TransformerModel for refined embeddings.

        Args:
            - transformer_model (nn.Module): An instance of the TransformerModel.
            - resume_embs (torch.Tensor): Original resume section embeddings.
              Shape: [batch_size, num_sections, dim].
            - job_desc_embs (torch.Tensor): Job description embeddings.
              Shape: [batch_size, dim].
            - positions (torch.Tensor): Positional indices for resume sections.
              Shape: [batch_size, num_sections].
            - entailment_scores (torch.Tensor): Precomputed entailment scores for attention.
              Shape: [batch_size, num_sections].

        Returns:
            torch.Tensor: The total loss value.
        """
        try:
            # Use the TransformerModel to compute refined embeddings
            refined_embs = transformer_model(
                resume_embs=resume_embs,
                job_desc_embs=job_desc_embs,
                positions=positions,
                entailment_scores=entailment_scores,
            )

            # 1. **Alignment Loss**:
            align_labels = torch.ones(
                refined_embs.size(0), device=device
            )  # Positive pairs
            align_loss = self.align_loss(refined_embs, job_desc_embs, align_labels)

            # 2. **Substance Loss** (if enabled):
            if self.include_substance_loss and self.substance_loss is not None:
                # Compute substance loss
                substance_loss = self.substance_loss(
                    refined_embs, resume_embs.mean(dim=1)
                )
                total_loss = align_loss + substance_loss
            else:
                total_loss = align_loss

            return total_loss
        except Exception as e:
            logger.error(f"Error in loss calculation: {e}")
            raise ValueError("Loss computation failed.") from e
