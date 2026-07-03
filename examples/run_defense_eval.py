"""Evaluate whether input-transformation defenses actually reduce attack success.

For each attack (FGSM, PGD, SimBA), this script:
1. Runs the attack normally (undefended) — baseline attack success.
2. Applies each defense to the adversarial image, then re-classifies.
3. Reports whether the defense "recovered" the correct prediction.
4. Also checks defense impact on CLEAN (non-attacked) accuracy — a defense
   that breaks normal images is useless regardless of attack performance.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advkit.attacks.fgsm import FGSM
from advkit.attacks.pgd import PGD
from advkit.attacks.blackbox import SimBA
from advkit.defenses.bit_depth import BitDepthReduction
from advkit.defenses.jpeg_compression import JPEGCompression
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DIR = Path("examples/sample_images")

def build_attacks(epsilon: float):
    return {
        "fgsm": lambda: FGSM(epsilon=epsilon),
        "pgd": lambda: PGD(epsilon=epsilon, alpha=0.005, num_steps=10),
        "blackbox": lambda: SimBA(epsilon=epsilon, max_queries=1000, early_stop=True),
    }

DEFENSES = {
    "bit_depth_4": lambda: BitDepthReduction(bits=4),
    "jpeg_75": lambda: JPEGCompression(quality=75),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defenses against attacks.")
    parser.add_argument("--image", type=str, default="sample.jpg")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Attack perturbation budget")
    args = parser.parse_args()

    image_path = SAMPLE_DIR / args.image
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return

    model = load_model(args.model)
    clean_tensor = load_image(image_path).unsqueeze(0)
    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))

    logger.info("=" * 78)
    logger.info("Defense Evaluation — Input Transformation Defenses")
    logger.info("=" * 78)
    logger.info("Clean prediction: class=%s confidence=%.4f", clean_label, clean_confidence)
    logger.info("-" * 78)

    # Sanity check: does each defense hurt CLEAN (non-attacked) accuracy?
    logger.info("Clean-image impact (defense applied to UNATTACKED image):")
    for defense_name, defense_fn in DEFENSES.items():
        defense = defense_fn()
        defended_clean = defense.apply(clean_tensor)
        defended_label, defended_confidence = predict(model, defended_clean.squeeze(0))
        status = "OK (label unchanged)" if defended_label == clean_label else "DEGRADED (label changed)"
        logger.info(
            "  %-14s -> class=%s confidence=%.4f [%s]",
            defense_name, defended_label, defended_confidence, status,
        )
    logger.info("-" * 78)

    results = []

    attacks = build_attacks(args.epsilon)
    for attack_name, attack_fn in attacks.items():
        attack = attack_fn()
        result = attack.generate(model, clean_tensor, clean_label)
        adversarial_tensor = result[0] if isinstance(result, tuple) else result

        adv_label, adv_confidence = predict(model, adversarial_tensor.squeeze(0))
        attack_succeeded = adv_label != clean_label

        logger.info("Attack: %s", attack_name)
        logger.info(
            "  Undefended -> class=%s confidence=%.4f [attack %s]",
            adv_label, adv_confidence, "SUCCEEDED" if attack_succeeded else "FAILED",
        )

        for defense_name, defense_fn in DEFENSES.items():
            defense = defense_fn()
            defended_tensor = defense.apply(adversarial_tensor)
            defended_label, defended_confidence = predict(model, defended_tensor.squeeze(0))
            recovered = defended_label == clean_label

            logger.info(
                "  + %-14s -> class=%s confidence=%.4f [%s]",
                defense_name, defended_label, defended_confidence,
                "RECOVERED" if recovered else "STILL FOOLED",
            )

            results.append({
                "attack": attack_name,
                "defense": defense_name,
                "attack_succeeded": attack_succeeded,
                "recovered": recovered,
            })
        logger.info("-" * 78)

    logger.info("Summary")
    logger.info("-" * 78)
    logger.info("%-10s %-14s %-18s %-15s", "Attack", "Defense", "Attack Success", "Defense Recovered")
    for r in results:
        logger.info(
            "%-10s %-14s %-18s %-15s",
            r["attack"], r["defense"],
            str(r["attack_succeeded"]), str(r["recovered"]),
        )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
