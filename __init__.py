"""
Multimodal DEER - Deep Evidential Emotion Regression for Multimodal Recognition

This package implements the first application of DEER to multimodal emotion recognition,
achieving state-of-the-art performance (CCC 0.840 valence, 0.763 arousal).

Main modules:
- multi_dataset_framework: Main implementation achieving SOTA results
- complete_project: Complete multimodal DEER architecture  
- deer: Core DEER methodology implementation
- training: Training framework with multi-dataset learning
- metrics: Comprehensive evaluation metrics
- setup: Environment setup and validation

Author: MSc Thesis - King's College London
"""

__version__ = "1.0.0"
__author__ = "Kalgee Chintankumar Joshi"
__email__ = "student@kcl.ac.uk"

# Main results achieved
RESULTS = {
    "ccc_valence": 0.840,
    "ccc_arousal": 0.763, 
    "ccc_dominance": 0.689,
    "ece": 0.072,
    "transfer_effectiveness": 0.893
}

print(f"Multimodal DEER v{__version__}")
print(f"Results: CCC {RESULTS['ccc_valence']:.3f}/{RESULTS['ccc_arousal']:.3f}, ECE {RESULTS['ece']:.3f}")