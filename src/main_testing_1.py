""" Temp for testing """

import os
from pipelines.resume_eval_pipeline import run_pipeline
from matching.resume_editing import edit_resp_to_match_req_wt_gpt


def main():
    """just for testing b/c I have to test from src dir"""

    resp_text = "Provided strategic insights to a major global IT vendor, optimizing their service partner ecosystem in Asia Pacific for improved local implementation outcomes."

    reqs_text1 = "Ability to form and refine hypotheses, gather supporting data, and make recommendations"
    reqs_text2 = "Excellent problem solving and analysis skills, including opportunity identification, market segmentation, and framing of complex/ambiguous problems"
    reqs_text3 = "Leverage first party and third party market data to build assets and programs that surface valuable insights to our business stakeholders and help inform product roadmaps"

    revised_text = edit_resp_to_match_req_wt_gpt(
        resp_id="1", resp_sent=resp_text, req_sent=reqs_text1
    )

    # Test with reqs_text1
    try:
        revised_text1 = edit_resp_to_match_req_wt_gpt(
            resp_id="1", resp_sent=resp_text, req_sent=reqs_text1
        )
        # print("Revised Text 1:", revised_text1)
    except Exception as e:
        print(f"An error occurred during revision 1: {e}")

    # Optionally, test with other requirement texts
    try:
        revised_text2 = edit_resp_to_match_req_wt_gpt(
            resp_id="2", resp_sent=resp_text, req_sent=reqs_text2
        )
        # print("Revised Text 2:", revised_text2)
    except Exception as e:
        print(f"An error occurred during revision 2: {e}")

    try:
        revised_text3 = edit_resp_to_match_req_wt_gpt(
            resp_id="3", resp_sent=resp_text, req_sent=reqs_text3
        )
        # print("Revised Text 3:", revised_text3)
    except Exception as e:
        print(f"An error occurred during revision 3: {e}")


if __name__ == "__main__":
    main()
