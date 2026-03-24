import numpy as np

def compare_models(classical_ratio, deep_heat):

    deep_ratio = np.mean(deep_heat > 0.6)

    agreement = 1 - abs(classical_ratio - deep_ratio)

    if agreement > 0.9:
        verdict = "Strong Agreement"
    elif agreement > 0.7:
        verdict = "Moderate Agreement"
    else:
        verdict = "Low Agreement"

    return {
        "deep_ratio": deep_ratio,
        "agreement": agreement,
        "verdict": verdict
    }