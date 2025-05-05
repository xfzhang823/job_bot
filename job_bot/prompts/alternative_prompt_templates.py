"""
filename: alternative_prompt_templates.py

Alternative templates advised by Claude

"""

SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT = """
You are a skilled professional at writing resumes. Please perform the following tasks:

<instructions>
1. Optimize the candidate text to increase semantic precision and similarity with the reference text.
2. Reduce the semantic distance between the candidate text and reference text.
3. Enhance entailment relationships, where the candidate text serves as the premise and the reference text serves as the hypothesis.
4. For the entailment task, the candidate text serves as the premise and the reference text serves as the hypothesis, reversing their typical roles.
5. Improve the overall alignment and relevance between the two texts, without compromising their directional relationship.
6. Do not copy specific experience durations directly from the reference text (e.g., "11 years experience in...", "more than 10 years in...", "8 years experience"). Instead, express relevant experience in more general terms or use ranges that encompass the candidate's actual experience.
</instructions>

<candidate_text>
{content_1}
</candidate_text>

<reference_text>
{content_2}
</reference_text>

<output_format>
Return only a valid JSON object, without markdown formatting, as follows:

{{
  "optimized_text": "Edited version of candidate text"
}}

Return only the JSON block without any additional text, explanations, or markdown syntax.
</output_format>
"""
