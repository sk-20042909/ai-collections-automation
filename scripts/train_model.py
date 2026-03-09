"""Convenience script – preprocess data and train models."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.preprocess import run_pipeline
from src.ml_models.train import train_all

run_pipeline()
train_all()
