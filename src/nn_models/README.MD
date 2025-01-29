# Transformer-Based Resume Alignment Module

This module provides components for a transformer-based system designed to align resumes and job descriptions effectively. It incorporates positional information into text embeddings and combines cross-attention with self-attention to model relationships between resumes and job descriptions.

## Features

1. **InputEncoder**:
   - Encodes input text using a pretrained BERT model.
   - Integrates positional information (log-scaled or dynamically learned) into embeddings.

2. **TransformerModel**:
   - Combines **cross-attention** (alignment with job descriptions) and **self-attention** (coherence within resume sections).
   - Uses entailment scores to guide attention mechanisms for improved alignment.

## Use Case
This module is particularly useful for tasks involving hierarchical data alignment, such as matching resumes to job descriptions, structured text alignment, or similar information retrieval tasks.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers library

### Install Required Packages
```bash
pip install torch transformers
```

---

## Components

### 1. **InputEncoder**
Encodes input text and integrates positional information for structured alignment tasks.

#### **Key Features**:
- **BERT Integration**: Leverages a pretrained BERT model for token embeddings.
- **Log Scaling**: Normalizes positional values using a logarithmic transformation.
- **Learned Positional Integration**:
  - **Learned Weighting**: Scales positional values with a single learnable weight.
  - **Learned Scaling**: Projects log-scaled positional values into high-dimensional vectors using a linear layer.

#### **Attributes**:
- `bert` (BertModel): Pretrained BERT model for generating embeddings.
- `tokenizer` (BertTokenizer): Tokenizer for preprocessing input text.
- `max_position` (int): Maximum positional value for normalization.
- `learned_weighting` (bool): Indicates whether to use learned weighting or learned scaling.
- `position_weight` (nn.Parameter, optional): Learnable parameter for scaling positional values.
- `positional_transform` (nn.Linear, optional): Linear layer for projecting positional values.

#### **Methods**:
- **`scale_positions(positions: torch.Tensor) -> torch.Tensor`**:
  Scales raw positional values using logarithmic scaling and reverses polarity.
- **`forward(texts: List[str], positions: torch.Tensor, max_length: int = 512) -> torch.Tensor`**:
  Encodes input text and integrates positional embeddings.

#### **Example Usage**:
```python
encoder = InputEncoder(model_name="bert-base-uncased", max_position=20000, learned_weight=True)
texts = ["Experience in software development.", "Knowledge of AI technologies."]
positions = torch.tensor([10001, 20002], dtype=torch.float32)
embeddings = encoder(texts, positions)
print(embeddings.shape)  # Output: [batch_size, hidden_size]
```

---

### 2. **TransformerModel**
A hybrid attention model that balances cross-attention (alignment) and self-attention (coherence) to process and align hierarchical data.

#### **Key Features**:
- **Cross-Attention**: Aligns resume embeddings with job descriptions.
- **Self-Attention**: Maintains coherence within resume sections.
- **Positional Embeddings**: Uses a learnable positional embedding layer for section-level order.
- **Trade-Off Parameter (`alpha`)**: Balances the influence of cross-attention and self-attention dynamically.

#### **Attributes**:
- `cross_attn` (nn.MultiheadAttention): Multi-head attention for cross-attention.
- `self_attn` (nn.MultiheadAttention): Multi-head attention for self-attention.
- `alpha` (nn.Parameter): Learnable parameter to balance cross-attention and self-attention.
- `positional_embed` (nn.Embedding): Positional embedding layer for maintaining section order.
- `output_layer` (nn.Linear): Final transformation layer for output.

#### **Methods**:
- **`forward(resume_embs: torch.Tensor, job_desc_embs: torch.Tensor, positions: torch.Tensor, entailment_scores: torch.Tensor) -> torch.Tensor`**:
  Combines cross-attention and self-attention to refine resume embeddings.

#### **Example Usage**:
```python
model = TransformerModel(dim=768)
resume_embs = torch.randn(32, 10, 768)  # [batch_size, num_sections, embedding_dim]
job_desc_embs = torch.randn(32, 768)  # [batch_size, embedding_dim]
positions = torch.randint(0, 10, (32, 10))  # [batch_size, num_sections]
entailment_scores = torch.randn(32, 10)  # [batch_size, num_sections]

output = model(resume_embs, job_desc_embs, positions, entailment_scores)
print(output.shape)  # Output: [batch_size, embedding_dim]
```

---

## Detailed Explanation

### **Positional Information**
Positional embeddings are critical for maintaining the order and structure of input sections (e.g., resume sections). This module supports two approaches:
1. **Learned Weighting**:
   - Scales log-normalized positional values using a single learnable parameter (`position_weight`).
   - Lightweight and interpretable.

2. **Learned Scaling**:
   - Projects log-normalized positional values into high-dimensional vectors using a linear layer (`positional_transform`).
   - More expressive and dynamic.

### **Hybrid Attention Mechanism**
The `TransformerModel` combines:
- **Cross-Attention**: Aligns job description embeddings (as queries) with resume embeddings (as keys and values).
- **Self-Attention**: Maintains coherence within resume sections by treating the sections as both queries and keys.
- **Entailment Scores**: Guides attention using precomputed entailment scores.

---

## Advantages
- **Flexibility**: Supports both lightweight and expressive positional embedding mechanisms.
- **Modularity**: Easily integrates with other pipelines for hierarchical data processing.
- **Adaptability**: Learnable parameters enable task-specific optimization.

---

## Contributing
Contributions are welcome! If you'd like to add features or fix issues, feel free to fork the repository and submit a pull request.

---

## License
This module is licensed under the MIT License. See the LICENSE file for details.

---

Let me know if you'd like to refine specific sections or include additional details!