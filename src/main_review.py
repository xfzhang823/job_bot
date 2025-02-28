from pipelines.resps_reqs_crosstab_pipeline import run_resps_reqs_crosstab_pipeline


def main():
    run_resps_reqs_crosstab_pipeline(score_threshold=0.0)  # Adjust threshold as needed


if __name__ == "__main__":
    main()
