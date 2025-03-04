from bert_score import score
from torch import Tensor

# Define the candidate and reference texts
candidate_text = "Accomplished database programming and data modeling expert with a track record of career advancement."
reference_text = "Experience with database programming and data modeling"

# Compute BERTScore precision
P: Tensor
P, R, F1 = score(
    [candidate_text],
    [reference_text],
    model_type="bert-base-uncased",
    lang="en",
    num_layers=9,
)

# Extract precision score
bert_score_precision = P.item()

# Print the result
print("BERTScore Precision:", bert_score_precision)
