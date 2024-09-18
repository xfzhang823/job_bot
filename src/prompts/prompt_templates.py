"""Prompt bank for standard prompt template/placeholders"""

# prompt_bank.py

CLEAN_JOB_PAGE_PROMPT = """task: keep content related to job roles, such as "About Us" sections, company descriptions, job responsibilities, job descriptions, requirements, qualifications, benefits, salary information, and company details. Remove any content that does not fall under these categories.
content: {content}
response format: text
return: filtered content only; do not include explanation
"""

# You need to use Escaping Curly Braces, "{{ }}", for JSON output examples!
# When use .format() method in Python strings, any curly braces {} in the string
# are treated as placeholders for variable substitution.
# Therefore, you need to escape them by doubling them ({{ for { and }} for })!!!

CONVERT_JOB_POSTING_TO_JSON_PROMPT = (
    "Extract structured data from the provided job description and convert it into a detailed, valid JSON format. "
    "Include all relevant sections such as 'Minimum qualifications', 'Responsibilities', 'Overview', and any other details. "
    "If the section names differ from the provided examples, map them to the closest relevant categories in the output JSON. "
    "For example, if a section is titled 'Desired Skills,' it could be mapped to 'Minimum qualifications.' If a section is missing, omit it from the JSON. "
    "Do not summarize; capture all content as-is in the 'content' section. Ensure the output is a well-formatted JSON object.\n\n"
    "Additionally, identify and include any time-based or experience-related data points (e.g., '11 years of experience in management consulting...') under their respective categories in the JSON output. "
    "These should not be omitted or summarized; they are crucial details to be captured as part of the structured data.\n\n"
    "Content: {content}\n\n"
    "Response format: JSON\n"
    "Output structure example:\n"
    "{{\n"
    '  "url": "<string>",\n'
    '  "job_title": "<string>",\n'
    '  "posted_date": "<string>",\n'
    '  "company": "<string>",\n'
    '  "content": {{\n'
    '    "overview": "<string>",\n'
    '    "minimum_qualifications": [\n'
    '      "<string>",\n'
    '      "11 years of hands-on experience with Python",\n'
    "      ...\n"
    "    ],\n"
    '    "responsibilities": [\n'
    '      "<string>",\n'
    "      ...\n"
    "    ],\n"
    '    "requirements": "<string>",\n'
    "    ...\n"
    "  }}\n"
    "}}\n\n"
    "Ensure every key and value is correctly enclosed in double quotes. Validate the JSON format before outputting it. "
    "Remove non-ASCII characters, special characters, and control characters. "
    "Trim leading and trailing spaces and newline characters from all values. "
    "Do not include any explanations or extraneous text in the output."
)

EXTRACT_JOB_REQUIREMENTS_PROMPT = (
    "You are a skilled professional at analyzing job descriptions. Given the following job description, extract all sections relevant to a candidate's qualifications, responsibilities, skills, education, experience, company culture, values, or any other unique criteria that are important to the employer. "
    "Search broadly for sections that match these categories, including those with different titles or phrasing that express similar ideas. Use synonyms and context to find all relevant information. Prioritize recall over precision to ensure no important details are missed.\n\n"
    "Job Description:\n{content}\n\n"
    "Instructions:\n"
    "1. Extract any text that provides information on what the employer wants, needs, or expects from candidates. This includes but is not limited to sections labeled as 'Qualifications,' 'Requirements,' 'Responsibilities,' 'Skills,' 'Education,' 'Experience,' 'About Us,' 'Company Overview,' 'Our Mission,' 'Core Values,' or similar.\n"
    "2. Combine all the extracted information into a single 'ask list' from the employer's perspective, including both technical and soft skills, educational and experience requirements, cultural fit, and any other relevant criteria.\n"
    "3. Categorize the items in this 'ask list' into five levels based on their degree of importance or realism:\n"
    "   - 'pie_in_the_sky': Aspirational qualifications or skills that represent the ideal candidate. These are nice-to-have attributes that few candidates are likely to possess (e.g., '10+ years of experience in a specific emerging technology,' 'Ph.D. in Data Science from a top-tier university').\n"
    "   - 'down_to_earth': Practical and achievable requirements that many qualified candidates would meet. These are commonly expected skills or experience levels (e.g., '3+ years of experience in a related field,' 'Proficiency in Excel and PowerPoint').\n"
    "   - 'bare_minimum': Essential qualifications or skills that are non-negotiable for the role. These are the minimum requirements a candidate must meet to be considered (e.g., 'Bachelor’s degree in relevant field,' 'Experience in customer service').\n"
    "   - 'cultural_fit': Attributes related to company culture, values, or soft skills that indicate a good match with the organization's environment and ethos (e.g., 'Strong team player,' 'Committed to diversity and inclusion').\n"
    "   - 'other': Any additional information that is relevant but does not fit into the above categories. Be creative and use your understanding of job descriptions to identify any unique or unusual requirements or expectations that the employer may have.\n"
    "{{\n"
    "  'pie_in_the_sky': [\n"
    "    '<Aspirational requirement 1>',\n"
    "    '<Aspirational requirement 2>',\n"
    "    '...'\n"
    "  ],\n"
    "  'down_to_earth': [\n"
    "    '<Practical requirement 1>',\n"
    "    '<Practical requirement 2>',\n"
    "    '...'\n"
    "  ],\n"
    "  'bare_minimum': [\n"
    "    '<Essential requirement 1>',\n"
    "    '<Essential requirement 2>',\n"
    "    '...'\n"
    "  ],\n"
    "  'cultural_fit': [\n"
    "    '<Cultural or values-based requirement 1>',\n"
    "    '<Cultural or values-based requirement 2>',\n"
    "    '...'\n"
    "  ],\n"
    "  'other': [\n"
    "    '<Other unique requirement 1>',\n"
    "    '<Other unique requirement 2>',\n"
    "    '...'\n"
    "  ]\n"
    "}}\n"
    "Ensure every key and value is correctly enclosed in double quotes. Validate the JSON format before outputting it. "
    "Remove non-ASCII characters, special characters, and control characters. "
    "Trim leading and trailing spaces and newline characters from all values. "
    "Do not include any explanations or extraneous text in the output."
)


# Per the new GPT model (o1) explanation:
# bullets are better for instructions and
# JSON should be used for output, esp. with examples.
# double {{ }} is needed for JSON output instruction.
SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Optimize the candidate text to increase semantic precision and similarity with the reference text.
2. Reduce the semantic distance between the candidate text and reference text.
3. Enhance entailment relationships, where the candidate text serves as the premise and the reference text serves as the hypothesis.
4. For the entailment task, the candidate text serves as the premise and the reference text serves as the hypothesis, reversing their typical roles.
5. Improve the overall alignment and relevance between the two texts, without compromising their directional relationship.

**Candidate text:**
"{content_1}"

**Reference text:**
"{content_2}"

**Return Format:**
Please return the result in JSON format as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Do not include any additional text or explanations.
"""

ENTAILMENT_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Enhance entailment relationships between **the premise text** and **the hypothesis text** by modifying ONLY the **premise text".
2. Improve the overall alignment and relevance between the two texts, without compromising their directional relationship.

**Premise text:**
"{content_1}"

**Hypothesis text:**
"{content_2}"

**Return Format:**
Please return the result in JSON format as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Do not include any additional text or explanations.
"""

SEMANTIC_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Optimize **the candidate text** to increase semantic precision and similarity with **the reference text**.
2. Reduce the semantic distance between **the candidate text** and the **reference text**.

**Candidate text:**
"{content_1}"

**Reference text:**
"{content_2}"

**Return Format:**
Please return the result in JSON format as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Do not include any additional text or explanations.
"""


STRUCTURE_TRANSFER_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Analyze the **source text** at a high level.
2. Apply the ""source text's** dependency structure to the **target text**. Ensure that the original meaning of the **target text** is mostly preserved.

**Target Text:**
"{content_1}"

**Source Text:**
"{content_2}"

**Return Format:**
Please return the result in JSON format as follows:

{{
  "optimized_text": "Edited version of the target text"
}}

Do not include any additional text or explanations.
"""
