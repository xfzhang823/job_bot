"""
TBA
"""

from utils.llm_data_utils import get_openai_api_key


# Analyze with LLM: WIP!!!
def analyze_resume_against_requirements(
    resume_json, requirements, model_id="gpt-4-turbo"
):
    """
    Analyzes the resume JSON against the job requirements and suggests modifications.

    Args:
        resume_json (dict): The JSON object containing the resume details.
        requirements (dict): The extracted requirements from the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        str: JSON-formatted suggestions for resume modifications.
    """
    get_openai_api_key()

    prompt = (
        f"Analyze the following resume JSON and identify sections that match the key job requirements. "
        f"Suggest modifications to better align the resume with the job description.\n\n"
        f"Resume JSON:\n{resume_json}\n\n"
        f"Key Requirements JSON:\n{requirements}\n\n"
        "Provide your suggestions in JSON format, with modifications highlighted."
    )

    # response = openai.chat.completions.create(
    #     model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.3
    # )

    # return response.choices[0].message.content
