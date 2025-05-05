"""
Filename: pipeline_manager.py
Helper class to organize pipelines according to which LLM provider to use (OpenAI, Claude, or LlaMa)

"""

# For now, I am not using this file, but keep it for now in case I expand the module.


class PipelineManager:
    def __init__(self, model_id, iteration_dirs):
        self.model_id = model_id
        self.iteration_dirs = iteration_dirs

    def run_preprocessing_pipeline(self):
        pass

    def run_evaluation_pipeline(self):
        pass

    def run_editing_pipeline(self):
        pass
