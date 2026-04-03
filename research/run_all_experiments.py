"""
Master experiment runner for SafeCommute AI research.
Runs all experiments sequentially and logs results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import traceback
from datetime import datetime

EXPERIMENTS = [
    ("Baseline", "research.experiments.baseline_eval"),
    ("Test-Time Augmentation", "research.experiments.test_time_augmentation"),
    ("Temperature Scaling", "research.experiments.temperature_scaling"),
    ("Curriculum Learning", "research.experiments.curriculum_training"),
    ("Feature Augmentation", "research.experiments.feature_augmentation"),
    ("Attention Variants", "research.experiments.attention_variants"),
    ("Depthwise Separable", "research.experiments.depthwise_model"),
    ("Knowledge Distillation", "research.experiments.distill_training"),
    ("Wav2Vec2 Features", "research.experiments.wav2vec2_features"),
    ("Contrastive Pretrain", "research.experiments.contrastive_pretrain"),
]


def main():
    print(f"{'='*60}")
    print(f"SafeCommute AI Research — All Experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    results = {}
    for name, module_path in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'='*60}")

        try:
            mod = __import__(module_path, fromlist=['main'])
            result = mod.main()
            results[name] = "SUCCESS"
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = f"FAILED: {e}"

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    # Also run ensemble after individual models are trained
    print(f"\n{'='*60}")
    print(f"  RUNNING ENSEMBLE (needs trained variants)")
    print(f"{'='*60}")
    try:
        from research.experiments import ensemble
        ensemble.main()
    except Exception as e:
        print(f"  Ensemble failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
