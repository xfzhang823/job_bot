import logging
from pathlib import Path
from torch import optim
from preprocessing.alignment_preprocessing import (
    load_data,
    scale_positions,
)

# From user defined
from nn_models.transformer_model import (
    InputEncoder,
    TransformerModel,
)
from nn_models.alignment_losses import AlignmentLoss
from training.transformer_training import (
    train_model,
    save_embeddings,
)

logger = logging.getLogger(__name__)


def run_alignment_with_transformer_pipelinepipeline():
    try:
        encoder = InputEncoder()
        model = TransformerModel().to(device="gpu")
        loss_fn = AlignmentLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # pylint: disable=E1101

        # Start training
        train_model("composite_scores.csv", "positions.csv", model, loss_fn, optimizer)

        # Save embeddings for later use
        save_embeddings(encoder, ["sample resume section 1", "sample resume section 2"])

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise RuntimeError("Pipeline execution failed.") from e
