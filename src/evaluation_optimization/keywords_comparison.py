"""keyword_comparison.py"""

import os
import sys
from dotenv import load_dotenv

import openai
from openai import OpenAI

from prompts.prompt_templates import MATCHING_KEYWORDS_AGAINST_TEXT_PROMPT

# Insert your content into the prompt
prompt = MATCHING_KEYWORDS_AGAINST_TEXT_PROMPT.format(
    content_1=content_1,
    content_2=json.dumps(resume_keywords, indent=2),
    content_3=json.dumps(job_description, indent=2),
)

# Send this formatted prompt to GPT (example with OpenAI API)
response = openai.Completion.create(engine="gpt-4", prompt=prompt, max_tokens=300)

print(response.choices[0].text)  # This will print the valid JSON response
