"""
Source code package for Fake News Detection Project.
"""

from .data_cleaning import clean_text, handle_missing_data, run_clean_pipeline

__all__ = ['clean_text', 'handle_missing_data', 'run_clean_pipeline']