"""Prompt bank for standard prompt template/placeholders"""

# prompt_bank.py

CLEAN_JOB_PAGE_PROMPT = """
Task: Filter job posting content to retain only the following elements:
- About Us / Company descriptions
- Job titles and descriptions
- Core responsibilities
- Required qualifications
- Benefits and compensation
- Company details and information

Content: {content}

Instructions:
- Remove all content not related to the categories above
- Return only the filtered text
- Do not include explanations or annotations

Format: Plain text output
"""


# !You need to use Escaping Curly Braces, "{{ }}", for JSON output examples!
# !When use .format() method in Python strings, any curly braces {} in the string
# !are treated as placeholders for variable substitution.
# !Therefore, you need to escape them by doubling them ({{ for { and }} for })!!!
# Updated this with Claude's suggested prompt (generally better than other LLMs)
CONVERT_JOB_POSTING_TO_JSON_PROMPT = """
You are a skilled professional at analyzing job descriptions. Extract key information and relevant content from the provided job description, converting it into a comprehensive, valid JSON format. Exclude purely website-related information or general company boilerplate not specific to the job.

Instructions:
1. Identify and extract the following key details if present:
   - url
   - Job title
   - Company name
   - Location
   - Salary information
   - Posted date (if available)
2. Capture the job-specific content, preserving its structure as much as possible.
3. Use clear section headers as keys in the JSON structure when present.
4. For content without clear section headers, use general keys like "description" or "additional_info".
5. Exclude information that is:
   - Purely related to website navigation or functionality
   - Generic company information not specific to the job role
   - Legal disclaimers or privacy policies unrelated to the job
   - Repetitive footer information

Content to process: {content}

Output structure:
{{
  "url": "<string or null if not found>",
  "job_title": "<string or null if not found>",
  "company": "<string or null if not found>",
  "location": "<string or null if not found>",
  "salary_info": "<string or null if not found>",
  "posted_date": "<string or null if not found>",
  "content": {{
    "<section_name>": "<full text of relevant section>",
    "<another_section>": "<full text of relevant section>",
    "description": "<any relevant content without a clear section header>",
    "additional_info": "<any remaining relevant content>"
  }}
}}

JSON Formatting Instructions:
1. Use "null" (without quotes) for any key details not found in the content.
2. Preserve newlines within string values using \\n.
3. Escape any double quotes within string values.
4. Ensure the output is a valid JSON object.

Extract all job-relevant information, but exclude website-specific content and generic company information not directly related to the job posting.
"""

EXTRACT_JOB_REQUIREMENTS_PROMPT = """
As a job description analyzer, extract and categorize qualifications, responsibilities, skills, and other criteria from \
the following job description. 
Ensure to:
- Include technical and soft skills, educational and experience requirements, and cultural fit aspects.
- Exclude any content related to benefits, perks, or compensation (e.g., health insurance, remote work options, \
401(k) matching, paid time off, bonuses, stock options.)

**Job Description:**
{content}

<Instructions:>
1. Extract all relevant information on employer expectations from sections like 'Qualifications,' 'Requirements,' 'Responsibilities,' 'Skills,' 'Education,' 'Experience,' 'About Us,' 'Company Overview,' 'Our Mission,' 'Core Values,' and similar.

2. Create an 'ask list' from the employer's perspective.

3. Categorize items into five levels, prioritizing within each category (1 being highest priority):
   - **pie_in_the_sky**: Aspirational, ideal qualifications (e.g., '10+ years in emerging tech,' 'Ph.D. from top-tier university')
   - **down_to_earth**: Practical, common requirements (e.g., '3+ years in related field,' 'Proficiency in Excel')
   - **bare_minimum**: Essential, non-negotiable qualifications (e.g., 'Bachelor's degree,' 'Customer service experience')
   - **cultural_fit**: Company culture and soft skills (e.g., 'Team player,' 'Committed to diversity')
   - **other**: Unique or unusual requirements

4. For quantitative requirements (e.g., years of experience, number of projects):
   - Include the specific quantity in the requirement description
   - If a range is given, use the lower bound for 'bare_minimum' and upper bound for 'down_to_earth' or 'pie_in_the_sky' as appropriate

5. **Extract all implicit requirements**: If the job description implies a skill or qualification \
(e.g., "collaborating with cross-functional teams" implies teamwork or communication skills), include it in \
the appropriate category.

6. Prioritize **completeness** over brevity. Ensure all relevant details are captured, even if they are not explicitly \
stated in labeled sections.   

</Instructions>

Output Structure:
{{
  "pie_in_the_sky": [
    "<Aspirational requirement 1>",
    "<Aspirational requirement 2>",
    ...
  ],
  "down_to_earth": [
    "<Practical requirement 1>",
    "<Practical requirement 2>",
    ...
  ],
  "bare_minimum": [
    "<Essential requirement 1>",
    "<Essential requirement 2>",
    ...
  ],
  "cultural_fit": [
    "<Cultural or values-based requirement 1>",
    "<Cultural or values-based requirement 2>",
    ...
  ],
  "other": [
    "<Other unique requirement 1>",
    "<Other unique requirement 2>",
    ...
  ]
}}

Ensure valid JSON format and return JSON block only, without any explanations, preambles, markdown syntax. Remove non-ASCII and special characters. Trim spaces and newlines.
"""


EXTRACT_JOB_REQUIREMENTS_PROMPT_OLD_VERSION = """
You are a skilled professional at analyzing job descriptions. Given the following job description, extract all sections relevant to a candidate's qualifications, responsibilities, skills, education, experience, company culture, values, or any other unique criteria that are important to the employer. Search broadly for sections that match these categories, including those with different titles or phrasing that express similar ideas. Use synonyms and context to find all relevant information. Prioritize recall over precision to ensure no important details are missed.

**Job Description:**
{content}

<Instructions:>
1. Extract any text that provides information on what the employer wants, needs, or expects from candidates. This includes but is not limited to sections labeled as 'Qualifications,' 'Requirements,' 'Responsibilities,' 'Skills,' 'Education,' 'Experience,' 'About Us,' 'Company Overview,' 'Our Mission,' 'Core Values,' or similar.
2. Combine all the extracted information into a single 'ask list' from the employer's perspective, including both technical and soft skills, educational and experience requirements, cultural fit, and any other relevant criteria.
3. Categorize the items in this 'ask list' into five levels based on their degree of importance or realism:
   - **pie_in_the_sky**: Aspirational qualifications or skills that represent the ideal candidate. These are nice-to-have attributes that few candidates are likely to possess (e.g., '10+ years of experience in a specific emerging technology,' 'Ph.D. in Data Science from a top-tier university').
   - **down_to_earth**: Practical and achievable requirements that many qualified candidates would meet. These are commonly expected skills or experience levels (e.g., '3+ years of experience in a related field,' 'Proficiency in Excel and PowerPoint').
   - **bare_minimum**: Essential qualifications or skills that are non-negotiable for the role. These are the minimum requirements a candidate must meet to be considered (e.g., 'Bachelor's degree in relevant field,' 'Experience in customer service').
   - **cultural_fit**: Attributes related to company culture, values, or soft skills that indicate a good match with the organization's environment and ethos (e.g., 'Strong team player,' 'Committed to diversity and inclusion').
   - **other**: Any additional information that is relevant but does not fit into the above categories. Be creative and use your understanding of job descriptions to identify any unique or unusual requirements or expectations that the employer may have.
</Instructions>

Output Structure:
{{
  "pie_in_the_sky": [
    "<Aspirational requirement 1>",
    "<Aspirational requirement 2>",
    "..."
  ],
  "down_to_earth": [
    "<Practical requirement 1>",
    "<Practical requirement 2>",
    "..."
  ],
  "bare_minimum": [
    "<Essential requirement 1>",
    "<Essential requirement 2>",
    "..."
  ],
  "cultural_fit": [
    "<Cultural or values-based requirement 1>",
    "<Cultural or values-based requirement 2>",
    "..."
  ],
  "other": [
    "<Other unique requirement 1>",
    "<Other unique requirement 2>",
    "..."
  ]
}}

JSON Formatting Instructions:
- Ensure every key and value is correctly enclosed in double quotes. 
- Validate the JSON format before outputting it. 
- Remove non-ASCII characters, special characters, and control characters. 
- Trim leading and trailing spaces and newline characters from all values. 
- Return only the JSON block without any explanations, preambles, markdown syntax, or other extraneous text in the output.
"""

SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Optimize the candidate text to increase semantic precision and similarity with the reference text.
2. Reduce the semantic distance between the candidate text and reference text.
3. Enhance entailment relationships, where the candidate text serves as the premise and the reference text serves as the hypothesis.
4. For the entailment task, the candidate text serves as the premise and the reference text serves as the hypothesis, reversing their typical roles.
5. Improve the overall alignment and relevance between the two texts, without compromising their directional relationship.
6. Do not copy specific experience durations directly from the reference text (e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). Instead, express relevant experience in more general terms or use ranges that encompass the candidate's actual experience.

**Candidate text:**
"{content_1}"

**Reference text:**
"{content_2}"

**Return Format:**
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
"""

ENTAILMENT_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Modify the **premise text** to enhance its entailment relationship with the **hypothesis text**.
2. Improve the overall alignment and relevance between the two texts without changing the directional relationship.
3. Do not copy specific experience durations directly from the reference text (e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). Instead, express relevant experience in more general terms or use ranges that encompass the candidate's actual experience.
4. Do not copy specific experience durations directly from the reference text (e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). Instead, express relevant experience in more general terms or use ranges that encompass the candidate's actual experience.

**Premise text:**
"{content_1}"

**Hypothesis text:**
"{content_2}"

**Return Format:**
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_text": "Edited version of premise text"
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
"""


SEMANTIC_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Optimize the **candidate text** to increase semantic precision and similarity with the **reference text**.
2. Reduce the semantic distance between them.
3. Do not copy specific experience durations directly from the reference text (e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). Instead, express relevant experience in more general terms or use ranges that encompass the candidate's actual experience.

**Candidate text:**
"{content_1}"

**Reference text:**
"{content_2}"

**Return Format:**
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
"""


STRUCTURE_TRANSFER_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Analyze the **source text** at a high level.
2. Apply the **source text's** sentence structure or syntactic dependencies to the **target text**, ensuring \
that the original meaning of the **target text** is preserved as much as possible.
3. If the source text has an active tone, retain it. Use strong action verbs such as 'led,' 'managed,' 'developed,' \
'executed,' etc., and explicitly avoid role-based constructions like 'professional who is,' 'was involved in,' or \
'held a position of.' Instead, focus on action-based descriptions that highlight what was achieved or done, \
such as 'led a team,' 'implemented strategies,' or 'drove improvements.'
3. Do not copy specific experience durations directly from the reference text \
(e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). \
Instead, express relevant experience in more general terms or use ranges that encompass the candidate's \
actual experience.

**Target Text:**
"{content_1}"

**Source Text:**
"{content_2}"

**Return Format:**
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_text": "Edited version of the target text"
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
"""

# Create a prompt and insert the JSON data into it
MATCHING_KEYWORDS_AGAINST_TEXT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:
1. Given the following **resume keywords** in JSON, extract keywords from the {content_1} sections.
2. Match these extracted keywords with the **job description** in JSON provided below, replace or modify \
the **resume keywords** if they are similar to **job description keywords**, and then prioritize them based on \
their importance to the job description.

**Resume keywords** in JSON:
{content_2}

**Job description** in JSON:
{content_3}

**Return Format:**
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_keywords": [],
  "prioritized_keywords": []
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
"""

# # Insert the JSON object as a string into the prompt
# formatted_prompt = prompt_template.format(resume_json=json.dumps(json_data, indent=2))
