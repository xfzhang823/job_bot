# **5. Training Loop**
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from nn_models.transformer_model import (
    InputEncoder,
    TransformerModel,
)
from nn_models.alignment_losses import AlignmentLoss


# Set logger
logger = logging.getLogger(__name__)


def train_step(
    resume_sections: List[str],
    job_desc: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    """
    Perform a single training step.

    Args:
        resume_sections (List[str]): The resume sections to encode.
        job_desc (str): The job description to encode.
        model (nn.Module): The model to train.
        loss_fn (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.

    Returns:
        float: The loss value for this step.
    """
    try:
        resume_embs = torch.stack([encoder.encode(s) for s in resume_sections]).to(
            device
        )
        job_emb = encoder.encode(job_desc).to(device)
        positions = torch.tensor(
            [position_map[i] for i in range(len(resume_sections))]
        ).to(device)

        # Forward pass
        refined = model(resume_embs, job_emb, positions)

        # Calculate loss
        loss = loss_fn(refined, resume_embs.mean(dim=0), job_emb)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    except Exception as e:
        logger.error(f"Error in training step: {e}")
        raise ValueError("Training step failed.") from e


# **6. Pipeline Orchestration**
def train_model(
    composite_scores_file: str,
    positions_file: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 10,
) -> None:
    """
    Orchestrate the training process, including loading data, training, and saving the model.

    Args:
        composite_scores_file (str): Path to the composite scores CSV file.
        positions_file (str): Path to the positions CSV file.
        model (nn.Module): The model to train.
        loss_fn (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        epochs (int): Number of epochs to train the model.
    """
    try:
        composite_scores, positions = load_data(composite_scores_file, positions_file)
        position_map = scale_positions(positions["responsibility_key"])

        for epoch in range(epochs):
            total_loss = 0
            for idx, row in composite_scores.iterrows():
                resume_sections = row[
                    "resume_sections"
                ]  # Assuming sections are preprocessed
                job_desc = row["job_description"]  # Job description text

                loss = train_step(resume_sections, job_desc, model, loss_fn, optimizer)
                total_loss += loss

            logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Save model and embeddings
        torch.save(model.state_dict(), "trained_model.pth")
        logger.info("Model saved!")

    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise RuntimeError("Model training failed.") from e


# **7. Saving Embeddings for Reuse**
def save_embeddings(
    encoder: InputEncoder, data: List[str], file_name: str = "embeddings.pt"
) -> None:
    """
    Save embeddings for the given data to a file for reuse.

    Args:
        encoder (InputEncoder): The encoder to use for generating embeddings.
        data (List[str]): The texts to encode and save.
        file_name (str): The file name to save the embeddings.
    """
    try:
        embeddings = []
        for text in data:
            embeddings.append(encoder.encode(text).cpu())
        embeddings_tensor = torch.stack(embeddings)
        torch.save(embeddings_tensor, file_name)
        logger.info(f"Embeddings saved to {file_name}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise RuntimeError("Failed to save embeddings.") from e
