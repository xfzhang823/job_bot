"""
File: text_similarity_finder
Author: XF Zhang
Last updated on:


Text Similarity Module

This module provides a `TextSimilarity` class that computes text similarity using various 
methods based on:
- BERT (Bidirectional Encoder Representations from Transformers) and 
- SBERT (Sentence-BERT).

BERT's architecture:
- bert-base-uncased: 12 layers (also called transformer blocks).
- bert-large-uncased: 24 layers.

The module uses the following similarity computation methods:

1. **Self-Attention Similarity**:
   - Computes the cosine similarity between the self-attention matrices of two texts.
   - Self-attention matrices capture the relationship between each word and every other word 
   in the input text, which is crucial for understanding contextual dependencies.
   - When we refer to the self-attention of an entire text window (i.e., the full input sequence) 
   in transformer models, we are referring to he last layer of the model.
   This method is sensitive to changes in word order and syntax but might not capture 
   nuanced differences in semantic meaning as effectively as other methods.

2. **Layer-Wise-Attention Similarity**:
   - Computes the cosine similarity between the self-attention matrices of two texts 
     for EACH LAYER of a transformer model (e.g., BERT).
   - Averages these similarities to produce a single aggregate score.
   
   This method captures both syntactic and semantic similarities by analyzing attention patterns 
   across all layers:
    - Lower layers focus on local relationships (e.g., word dependencies), 
    - higher layers capture more abstract, global relationships (e.g., topic coherence). 
    - Averaging these similarities provides a comprehensive measure of how similarly two texts are 
   processed by the model -> useful for tasks like paraphrase detection and semantic similarity.

3. **Hidden State Similarity**:
   - Computes the cosine similarity between the hidden states (representations) of two texts in BERT.
   - Hidden states from the last layer of BERT capture contextualized word embeddings. 
   
   This method captures semantic relationships but can be influenced by subtle changes in wording. 
   It is generally more effective than self-attention for capturing meaning but still relies on 
   token-level similarities.

4. **[CLS] Token Embedding Similarity** (CLS stands for classification):
   - Computes the cosine similarity between the [CLS] token embeddings of two texts. 
   - The [CLS] token is a special token added at the beginning of every input sequence, and its 
   corresponding hidden state in the final layer is used as a pooled representation 
   of the entire sequence. 
   
   This method is commonly used for classification tasks and provides a high-level summary of 
   the input's meaning, making it effective for sentence-level similarity. However, it might miss 
   finer nuances if the sentences are complex.

5. **SBERT Sentence Embedding Similarity**:
   - Computes the cosine similarity between the sentence embeddings generated by SBERT (Sentence-BERT).
   - SBERT is fine-tuned specifically for generating semantically meaningful sentence embeddings. 
   It uses a modified BERT architecture with pooling layers and is highly effective for capturing 
   semantic similarities between sentences. 
   
   This method tends to outperform the standard BERT-based methods when it comes to 
   natural language understanding tasks that require semantic matching, paraphrase identification, 
   or sentence clustering.

6. **Semantic Textual Similarity (STS)**:
   - STS measures the degree to which two pieces of text (sentences, phrases, or paragraphs) 
   express the same meaning or ideas. 
   - The STS score ranges from 0 (completely dissimilar) to 1 (completely similar) &
   provides a nuanced understanding of text similarity by incorporating deep semantic insights.
   
   This method effectively captures complex semantic relationships such as 
   negation, emphasis, paraphrasing, synonymy, and antonymy, making it useful for 
   applications that require a deeper understanding of meaning.
 

Each method provides unique insights into the text's structure, syntax, and meaning, 
making the `TextSimilarity` class a comprehensive tool for exploring text similarities 
using transformer-based models.

Usage:
    To use the module, 
    - create an instance of the `TextSimilarity` class and 
    - call its methods with appropriate text inputs.
"""

# Dependencies
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util


def convert_dict_to_array(similarity_dict):
    """
    Convert a dictionary of similarity scores to a list (array).

    Args:
        similarity_dict (dict): A dictionary containing similarity scores.

    Returns:
        list: A list of similarity scores.
    """
    return list(similarity_dict.values())


class TextSimilarity:
    """
    A class to compute text similarities using various methods, including BERT-based and SBERT-based approaches.
    """

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        sbert_model_name="all-MiniLM-L6-v2",
        sts_model_name="stsb-roberta-base",
    ):
        """
        Initialize models and tokenizers for BERT, SBERT, and STS.
        """
        # Load BERT models and tokenizer
        self.bert_model_attention = BertModel.from_pretrained(
            bert_model_name, output_attentions=True, attn_implementation="eager"
        )  # attn_implementation="eager" ->
        # the code is future-proof for Transformers v5.0.0 & beyond (the default PyTorch
        # implementation will no longer automatically fall back to the manual implementation.)
        self.bert_model_hidden_states = BertModel.from_pretrained(
            bert_model_name, output_hidden_states=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load SBERT model
        self.sbert_model = SentenceTransformer(sbert_model_name)

        # Load STS model
        self.sts_model = SentenceTransformer(sts_model_name)

    def get_attention(self, input_text, context=None):
        """
        Get the attention scores for a given text using BERT.

        Attention scores provide insight into which words in a sentence are focusing on which other words.
        This method is useful for understanding how different words influence each other in a given text.
        """

        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute attention scores
        input = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        output = self.bert_model_attention(**input)

        # Extract attention scores from the output
        attentions = output.attentions  # A list of tensors for each layer

        return attentions

    def pad_to_match(self, tensor1, tensor2):
        """
        Pad two tensors to have the same dimensions.

        This is useful when comparing tensors of different sizes, such as attention matrices
        or hidden states. Padding ensures that the cosine similarity computation is performed
        on tensors of the same shape.
        """
        # Ensure inputs are tensors
        if isinstance(tensor1, tuple):
            tensor1 = tensor1[0]  # Convert tuple to tensor if necessary
        if isinstance(tensor2, tuple):
            tensor2 = tensor2[0]  # Convert tuple to tensor if necessary

        # Determine the max size in both dimensions (seq_len)
        max_size_0 = max(
            tensor1.size(0), tensor2.size(0)
        )  # Number of rows (sequence length)
        max_size_1 = max(
            tensor1.size(1), tensor2.size(1)
        )  # Number of columns (sequence length)

        # Pad tensor1 if needed
        pad_tensor1 = (0, max_size_1 - tensor1.size(1), 0, max_size_0 - tensor1.size(0))
        tensor1 = F.pad(tensor1, pad_tensor1, value=0)

        # Pad tensor2 if needed
        pad_tensor2 = (0, max_size_1 - tensor2.size(1), 0, max_size_0 - tensor2.size(0))
        tensor2 = F.pad(tensor2, pad_tensor2, value=0)

        return tensor1, tensor2

    def layer_wise_attention_similarity(self, text1, text2, context=None):
        """
        Compute the average cosine similarity of attention matrices across all layers
        of BERT for two texts.

        This method captures both syntactic and semantic similarities by analyzing
        attention patterns at different layers of the model.
        """
        # Get attentions for both texts
        attentions1 = self.get_attention(text1, context)
        attentions2 = self.get_attention(text2, context)

        # Initialize list to store cosine similarities for each layer
        layer_similarities = []

        # Iterate over layers and compute cosine similarity for each
        for layer_attention1, layer_attention2 in zip(attentions1, attentions2):
            # Mean over heads for each layer
            # (batch_size, num_heads, seq_len, seq_len)
            # -> (batch_size, seq_len, seq_len)
            attention1_mean = layer_attention1.mean(dim=1).squeeze(0)
            attention2_mean = layer_attention2.mean(dim=1).squeeze(0)

            # Pad attention matrices to have the same size
            attention1_mean, attention2_mean = self.pad_to_match(
                attention1_mean, attention2_mean
            )

            # Flatten attention matrices to compare
            attention1_flat = attention1_mean.view(-1)  # Flatten to a 1D tensor
            attention2_flat = attention2_mean.view(-1)  # Flatten to a 1D tensor

            # Compute cosine similarity for the current layer
            cosine_sim = torch.nn.functional.cosine_similarity(
                attention1_flat, attention2_flat, dim=0
            )
            layer_similarities.append(cosine_sim.item())

        # Compute the average similarity across all layers
        average_similarity = sum(layer_similarities) / len(layer_similarities)
        return average_similarity

    def self_attention_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the attention matrices of two texts.

        This method compares the attention patterns between two sentences, which can reveal syntactic
        similarities but may not fully capture semantic differences.
        """
        # Get attentions for both texts
        attentions1 = self.get_attention(text1, context)
        attentions2 = self.get_attention(text2, context)

        # Extract the last layer's attention for both texts
        attention1 = (
            attentions1[-1].mean(dim=1).squeeze(0)
        )  # Mean over heads, shape (seq_len, seq_len)
        attention2 = attentions2[-1].mean(dim=1).squeeze(0)

        # Pad attention matrices to have the same size
        attention1, attention2 = self.pad_to_match(attention1, attention2)

        # Flatten attention matrices to compare
        attention1_flat = attention1.view(-1)  # Flatten to a 1D tensor
        attention2_flat = attention2.view(-1)  # Flatten to a 1D tensor

        # Ensure both tensors have the same size after padding and flattening
        assert (
            attention1_flat.size() == attention2_flat.size()
        ), "Padded tensors must have the same size."

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            attention1_flat, attention2_flat, dim=0
        )

        return cosine_sim

    def get_hidden_states(self, input_text, context=None):
        """
        Get the hidden states for a given text using BERT.
        """
        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute hidden states
        inputs = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.bert_model_hidden_states(**inputs)

        # Extract hidden states from the output (last hidden state)
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)

        return hidden_states.squeeze(0)  # Remove batch dimension if batch_size = 1

    def self_hidden_state_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the hidden states of two texts.
        """
        # Get hidden states for both texts
        hidden1 = self.get_hidden_states(text1, context)
        hidden2 = self.get_hidden_states(text2, context)

        # Pad hidden states to have the same size
        hidden1, hidden2 = self.pad_to_match(hidden1, hidden2)

        # Flatten hidden states to compare
        hidden1_flat = hidden1.view(-1)  # Flatten to a 1D tensor
        hidden2_flat = hidden2.view(-1)  # Flatten to a 1D tensor

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            hidden1_flat, hidden2_flat, dim=0
        )
        return cosine_sim

    def get_cls_embedding(self, input_text, context=None):
        """
        The [CLS] embedding is the embedding of the [CLS] token from the last hidden state of the BERT model.
        The [CLS] token is a special token added at the beginning of every input sequence,
        and its corresponding hidden state in the final layer is often used as a pooled representation of
        the entire sequence for classification and other sequence-level tasks (the final state.)

        IT DOES NOT NEED PADDING BECAUSE IT'S THE LAST LAYER ONLY!

        CLS stands for classification.
        """
        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute hidden states
        inputs = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.bert_model_hidden_states(**inputs)

        # Extract the [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: (batch_size, hidden_size)

        return cls_embedding.squeeze(0)  # Remove batch dimension if batch_size = 1

    def cls_embedding_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the [CLS] token embeddings of two texts.

        CLS stands for classification.
        """
        # Get [CLS] token embeddings for both texts
        cls1 = self.get_cls_embedding(text1, context)
        cls2 = self.get_cls_embedding(text2, context)

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(cls1, cls2, dim=0)

        return cosine_sim

    def get_sentence_embedding(self, text):
        """
        Get the sentence embedding for a given text using SBERT (Sentence-BERT).
        """
        # Compute the embedding
        embedding = self.sbert_model.encode(text, convert_to_tensor=True)
        return embedding

    def sbert_similarity(self, text1, text2):
        """
        Compute the cosine similarity between sentence embeddings using SBERT (Sentence-BERT).
        """
        # Get embeddings for both texts
        embedding1 = self.get_sentence_embedding(text1)
        embedding2 = self.get_sentence_embedding(text2)

        # Compute cosine similarity
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def sts_similarity(self, text1, text2):
        """
        Compute the semantic textual similarity (STS) between two texts using
        a fine-tuned model.

        More sensitive to subtle differences in meaning, STS measures the degree to
        which two texts express the same meaning or ideas.
        """
        # Get embeddings for both texts
        embedding1 = self.sts_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sts_model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def all_similarities(self, text1, text2, context=None):
        """
        Compute all similarity scores for a given pair of texts using various methods.

        Returns:
            dict: A dictionary containing similarity scores computed using different methods.
        """
        similarities = {}

        # Compute Self-Attention Similarity
        similarities["self_attention_similarity"] = self.self_attention_similarity(
            text1, text2, context
        ).item()

        # Compute Layer-Wise Attention Similarity
        similarities["layer_wise_attention_similarity"] = (
            self.layer_wise_attention_similarity(text1, text2, context)
        )

        # Compute Hidden State Similarity
        similarities["self_hidden_state_similarity"] = (
            self.self_hidden_state_similarity(text1, text2, context).item()
        )

        # Compute [CLS] Token Embedding Similarity
        similarities["cls_embedding_similarity"] = self.cls_embedding_similarity(
            text1, text2, context
        ).item()

        # Compute SBERT Sentence Embedding Similarity
        similarities["sbert_similarity"] = self.sbert_similarity(text1, text2)

        # Compute Semantic Textual Similarity (STS)
        similarities["sts_similarity"] = self.sts_similarity(text1, text2)

        return similarities

    def print_tensor(self, tensor):
        """
        Print a tensor in a formatted way.
        """
        for row in tensor:
            print(" ".join(f"{x:.2f}" for x in row))


def main():
    """
    Main function for testing the similarity methods.
    """

    # # Example context and texts
    # context = "John is required to be at the event because he is the main speaker."
    # text1 = "John is present at the event."
    # text2 = "John is absent at the event."

    # # Example from Meta AI
    # context = "Embedded software development, memory management, C++"
    # text1 = """
    # Relying solely on C++ atomics and concurrency features is sufficient for ensuring thread-safe access
    # to shared data and protecting critical sections in multi-threaded embedded systems, providing a portable
    # and efficient solution.
    # """
    # text2 = """
    # While C++ atomics and concurrency features provide a foundation for thread safety,
    # they are insufficient on their own for ensuring data integrity and consistency in multi-threaded
    # embedded systems, and must be supplemented with platform-specific optimizations and custom solutions to
    # address the unique challenges of resource-constrained environments.
    # """

    # # Example context and texts from GPT-4o
    # context = "Artificial Intelligence in Healthcare, data privacy, machine learning models, patient data security."
    # text1 = """
    # Implementing standard machine learning models in healthcare applications, along with widely adopted data privacy
    # protocols such as encryption and access control, is generally sufficient for ensuring patient data security. These
    # methods provide a reliable foundation for preventing unauthorized access and maintaining data confidentiality in most scenarios.
    # """
    # text2 = """
    # While standard machine learning models and commonly used data privacy protocols like encryption and access control provide
    # a baseline level of security in healthcare applications, they are not sufficient on their own. Specialized approaches, such as
    # differential privacy and secure multi-party computation, are needed to address the unique challenges of protecting sensitive
    # patient data in complex machine learning environments.
    # """

    # Example resume bullet and job requirements
    context = "Artificial Intelligence in Healthcare, data privacy, machine learning models, patient data security."
    text1 = """
    Developed Python tools to automate and accelerate internal processes, cutting report preparation and data analysis time by over 40%.
    """
    text2 = """
    11 years of experience in management consulting, product management and strategy, or analytics in a technology company.
    Experience working with and analyzing data, and managing multiple cross-functional programs or projects.
    Experience with performing market analysis and developing competitive intelligence.
    Ability to manage executive stakeholders and communicate with a highly technical management team.
    Ability to form and refine hypotheses, gather supporting data, and make recommendations.
    Excellent problem solving and analysis skills, including opportunity identification, market segmentation, and framing of complex/ambiguous problems.
    """

    # Print context, text1, text2
    print(f"Context: {context}\nText 1: {text1}\nText 2: {text2}\n")
    # Create an instance of the TextSimilarity class
    text_similarity = TextSimilarity()

    # Compute similarity using self-attention
    cosine_sim = text_similarity.self_attention_similarity(
        text1, text2, context=context
    )
    print(f"Cosine Similarity between attentions: {cosine_sim.item():.4f}")

    # Compute similarity using layer-wise attention analysis
    layer_attention_similarity = text_similarity.layer_wise_attention_similarity(
        text1, text2, context=context
    )
    print(
        f"Layer-Wise Attention Similarity (average across layers): {layer_attention_similarity:.4f}"
    )

    # Compute similarity using hidden states
    cosine_sim = text_similarity.self_hidden_state_similarity(
        text1, text2, context=context
    )
    print(f"Cosine Similarity between hidden states: {cosine_sim.item():.4f}")

    # Compute similarity using CLS embedding
    cosine_sim = text_similarity.cls_embedding_similarity(text1, text2, context=context)
    print(f"Cosine Similarity between [CLS] embeddings: {cosine_sim.item():.4f}")

    # Compute similarity using SBERT
    cosine_sim = text_similarity.sbert_similarity(text1, text2)
    print(f"Cosine Similarity using SBERT: {cosine_sim:.4f}")

    # Compute similarity using STS
    sts_similarity = text_similarity.sts_similarity(text1, text2)
    print(f"Cosine Similarity using STS: {sts_similarity:.4f}")


if __name__ == "__main__":
    main()