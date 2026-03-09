"""Convenience script – generate the synthetic dataset only."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset_generator.generate import save_dataset
save_dataset()
