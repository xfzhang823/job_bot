"""
transformer_model.py
Author: Xiao-Fei Zhang
Last upated: 2025 Jan


This module implements components for a transformer-based system designed for 
resume and job description alignment. It includes:

1. **InputEncoder**: Encodes input text using a pretrained BERT model and integrates 
   positional information (learned or scaled).

2. **TransformerModel**: Combines cross-attention (alignment with job descriptions) 
   and self-attention (coherence within resume sections), guided by entailment scores.

The system is designed for tasks requiring structured alignment between hierarchical 
data, such as resumes and job descriptions.

Classes:
    - InputEncoder: Encodes text with positional embeddings.
    - TransformerModel: A hybrid attention model for alignment and coherence.

Usage:
    This module can be used as part of a pipeline for processing resumes and 
    job descriptions. 
    Example usage is embedded in the respective classes.
"""

# Dependencies
import logging
from typing import List, Tuple
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# *Input Encoder (Embedding)
class InputEncoder(nn.Module):
    """
    Encodes input text using a pretrained BERT model and integrates positional information.

    This class supports two approaches for incorporating positional information:
    *1. Learned Weighting: Applies a single learnable weight to scale positions before
       *adding them to the text embeddings.
    *2. Learned Scaling: Transforms log-scaled positional values into a high-dimensional
       *vector using a feedforward layer.

    Attributes:
        - bert (BertModel): Pretrained BERT model for generating text embeddings.
        - tokenizer (BertTokenizer): Tokenizer for preprocessing input text.
        - max_position (int): Maximum position value for log scaling.
        - learned_weighting (bool): Indicates whether to use learned weighting or
        learned scaling.
        - position_weight (nn.Parameter, optional): Learnable weight for
        scaling positional values (used if learned_weighting=True).
        - positional_transform (nn.Linear, optional): Linear transformation for
        learned scaling (used if learned_weighting=False).

    Methods:
        - scale_positions: Scales raw positional values using logarithmic scaling.
        - forward: Encodes input text and integrates positional embeddings.

    Args:
        - model_name (str): The name of the pretrained BERT model
        (default: "bert-base-uncased").
        - max_position (int): Maximum position value for scaling (default: 20000).
        - learned_weight (bool): Whether to use learned weighting (True)
        or learned scaling (False).

    Example:
        encoder = InputEncoder(model_name="bert-base-uncased",
                                max_position=20000,
                                learned_weight=True)
        texts = ["Experience in software development.", "Knowledge of AI technologies."]
        positions = torch.tensor([10001, 20002], dtype=torch.float32)
        embeddings = encoder(texts, positions)
        print(embeddings.shape)  # Output: [batch_size, hidden_size]
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_position: int = 20000,
        learned_weight: bool = True,
    ) -> None:
        """
        Initialize the encoder with a pretrained BERT model.

        Args:
            - model_name (str): The name of the pretrained model.
            - max_position (int): Maximum position value for scaling.
            - learned_weighting (bool): Whether to use learned weighting (True) or
            learned scaling (False).
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)  # Load pretrained BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_position = max_position  # Used for log scaling
        self.learned_weighting = learned_weight

        # Add a learned weight to scale position further
        if self.learned_weighting:
            # Define a single learnable weight (alpha) for learned weighting
            self.position_weight = nn.Parameter(
                torch.tensor(0.5)
            )  # Initialize with 0.5
        else:
            # Define a linear transformation layer for learned scaling
            self.positional_transform = nn.Linear(1, self.bert.config.hidden_size)

    def scale_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply logarithmic scaling to position values.

        Args:
            positions (torch.Tensor): The raw position values (1D tensor).

        Returns:
            torch.Tensor: Log-scaled position values between 0 and 1.
        """
        scaled_positions = torch.log(positions + 1) / torch.log(
            torch.tensor(self.max_position + 1, dtype=torch.float32)
        )

        # Reverse polarity (e.g., smaller positions map to higher importance)
        return 1 - scaled_positions

    def forward(
        self, texts: List[str], positions: torch.Tensor, max_length: int = 512
    ) -> torch.Tensor:
        """
        Forward pass to encode a list of texts with positional embeddings.

        Args:
            - texts (List[str]): The texts to encode.
            - positions (torch.Tensor): Positional encodings for the sections.
            - max_length (int): Maximum length for padding/truncation.

        Returns:
            torch.Tensor: Encoded text representations with positional embeddings.
        """
        try:
            # Tokenize input texts
            encoded = self.tokenizer(
                texts,
                padding=True,  # Pad sequences to the same length
                truncation=True,  # Truncate sequences to max_length
                max_length=max_length,  # Maximum allowed sequence length
                return_tensors="pt",  # Return PyTorch tensors
            ).to(
                device
            )  # `encoded` is a dictionary of tensors, not a tensor itself

            # Get BERT outputs
            outputs = self.bert(
                **encoded
            )  # `outputs` is an object containing multiple tensors, not a tensor itself

            # Extract hidden states for all tokens from the final layer
            text_embeddings: torch.Tensor = (
                outputs.last_hidden_state
            )  # Shape: [batch_size, seq_len, hidden_size]

            # Extract [CLS] token embeddings (the first token for each sequence)
            cls_text_embeddings: torch.Tensor = text_embeddings[
                :, 0, :
            ]  # Shape: [batch_size, hidden_size], summarizing the sequence

            # * CLS is the embedding of the [CLS] token in models like BERT:
            # * It represents the entire document, sentence, or text block, and is computed by
            # * combining all token embeddings through the self-attention mechanism,
            # * making it a summary of the input sequence for tasks like classification.

            # : (first position): Select all examples in the batch.
            # 0 (second position): Select the first token for each sequence in each batch
            # (the [CLS] token in BERT).
            # : (third position): Select all dimensions of the embedding vector for
            # the selected token (of size hidden_size).

            # Scale positions and expand to match embedding dimensions [x1+pos, x2+pos, ...]
            scaled_positions = self.scale_positions(positions)  # Shape: [batch_size]

            if self.learned_weighting:
                # Learned Weighting: Scale positions by learnable alpha and add to CLS embeddings
                combined_embeddings = (
                    cls_text_embeddings
                    + self.position_weight * scaled_positions.unsqueeze(-1)
                )
            else:
                # Learned Scaling: Transform scaled positions into high-dimensional vectors
                scaled_positions = scaled_positions.unsqueeze(
                    -1
                )  # Shape: [batch_size, 1]
                scaled_position_embeddings = self.positional_transform(scaled_positions)
                combined_embeddings = cls_text_embeddings + scaled_position_embeddings

            return combined_embeddings

        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise ValueError("Failed to encode texts.") from e


# *Hybrid Attention Model
class TransformerModel(nn.Module):
    """
    A model that uses standard attention with a learnable alpha parameter to balance
    cross-attention (alignment with the job description) and self-attention
    (coherence within resume sections), while integrating precomputed entailment scores to
    guide the attention mechanism.

    Args:
        dim (int): The embedding dimension for each token (e.g., 768 for BERT embeddings).
    """

    def __init__(self, dim: int = 768) -> None:
        super().__init__()

        # Cross-attention (job description as query, resume sections as
        # key and value)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)

        # Self-attention (coherence within resume sections)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=8)

        # Learnable trade-off parameter (alpha) between cross-attention and
        # self-attention
        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )  # Initialized to 0.5, but will be learned

        # Positional encoding for resume sections (helps maintain the order
        # of sections)
        self.positional_embed = nn.Embedding(
            10, dim
        )  # Assume a max of 10 sections in the resume

        # Final transformation to obtain the output after attention
        self.output_layer = nn.Linear(dim, dim)

    def forward(
        self,
        resume_embs: torch.Tensor,
        job_desc_embs: torch.Tensor,
        positions: torch.Tensor,
        entailment_scores: torch.Tensor,
        return_attn_scores: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass of the model:
        - using both cross-attention (for alignment) and self-attention (for coherence),
        - while integrating entailment scores to adjust attention.

        Args:
            - resume_embs (torch.Tensor):
            Embeddings for resume sections, shape [batch_size, num_sections, dim].
            - job_desc_embs (torch.Tensor):
            Embeddings for the job description, shape [batch_size, dim].
            - positions (torch.Tensor):
            Positional encodings for each section, shape [batch_size, num_sections].
            - entailment_scores (torch.Tensor):
            Precomputed entailment scores, shape [batch_size, num_sections].
            - return_attn_scores (bool): Whether to return attention scores
            (default: False).

        Returns:
            torch.Tensor: The refined embedding of the resume sections, shape
            [batch_size, dim].
        """
        try:
            # Add positional embeddings to resume sections
            positions = positions.unsqueeze(-1)  # Shape: [batch_size, num_sections, 1]
            resume_embs = resume_embs + self.positional_embed(positions)

            # Normalize entailment scores to use as attention weights
            attention_weights = torch.softmax(
                entailment_scores, dim=-1
            )  # Normalize to [0, 1]

            # Cross-attention: job description as query, resume sections as key and value
            cross_out, cross_attn_scores = self.cross_attn(
                query=job_desc_embs.unsqueeze(0), key=resume_embs, value=resume_embs
            )

            # Apply entailment-based attention weights to cross-attention output
            cross_out = cross_out * attention_weights.unsqueeze(
                -1
            )  # Scale cross-attention output by entailment scores

            # Self-attention: resume sections as query, key, and value
            self_out, self_attn_scores = self.self_attn(
                query=resume_embs, key=resume_embs, value=resume_embs
            )

            # Combine both attention outputs using alpha (the trade-off parameter)
            combined = self.alpha * cross_out + (1 - self.alpha) * self_out.mean(dim=0)

            # Final transformation
            refined_embs = self.output_layer(combined)

            if return_attn_scores:
                return refined_embs, cross_attn_scores, self_attn_scores

            return refined_embs
        except Exception as e:
            # Error handling for debugging
            logger.error(f"Error in forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e
