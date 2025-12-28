import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from preprocessing_severity_class import PreprocessingSeverity

class InferencePreprocessor(PreprocessingSeverity):
    """
    Subclass of PreprocessingSeverity adapted for inference on Traffic Data.
    """
    def __init__(self):
        super().__init__()
        # Load processors immediately
        self.load_processors()
        
    def preprocess_inference(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for inference (Wrapper for parent method).
        """
        # Call parent's method which now handles the traffic logic correctly
        return super().preprocess_inference(df_input)
