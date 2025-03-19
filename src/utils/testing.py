import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load RoBERTa and DeBERTa models
roberta_model_name = "roberta-large-mnli"
deberta_model_name = "microsoft/deberta-large-mnli"

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)

deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name)


def get_entailment_score(model, tokenizer, premise, hypothesis):
    """Compute entailment score for a given premise and hypothesis."""
    inputs = tokenizer.encode_plus(
        premise, hypothesis, return_tensors="pt", truncation=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[
            0
        ]  # Convert logits to probabilities
    return probs[2]  # Entailment probability


# Sample test cases (from generated test file)
test_cases = [
    (
        "Developed strategic insights and market analysis for enterprise software clients, helping them optimize go-to-market strategies.",
        "Experience conducting market analysis and providing strategic recommendations.",
    ),
    (
        "Led a team of data scientists in building predictive models for healthcare analytics.",
        "Experience in managing teams and working with healthcare data.",
    ),
    (
        "Designed and implemented cloud infrastructure solutions for financial services.",
        "Experience in pharmaceutical data analytics.",
    ),
    (
        "Created reports, blogs, and presentations summarizing competitive intelligence insights.",
        "Responsible for storytelling with data and structuring strategic insights.",
    ),
    (
        "Managed a global portfolio of Fortune 500 clients in the software industry, providing analytics-driven recommendations.",
        "Experience working with large clients and delivering data-driven insights.",
    ),
    (
        "Published research papers on neural networks and deep learning architectures.",
        "Proven experience in customer success and relationship management.",
    ),
]

# Run test cases through both models
for premise, hypothesis in test_cases:
    roberta_score = get_entailment_score(
        roberta_model, roberta_tokenizer, premise, hypothesis
    )
    deberta_score = get_entailment_score(
        deberta_model, deberta_tokenizer, premise, hypothesis
    )

    print(f"Premise: {premise}\nHypothesis: {hypothesis}")
    print(f"RoBERTa Entailment Score: {roberta_score:.4f}")
    print(f"DeBERTa Entailment Score: {deberta_score:.4f}")
    print("-" * 80)
