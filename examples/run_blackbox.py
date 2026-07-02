from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advkit.attacks.blackbox import SimBA
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image, tensor_to_image
from advkit.utils.metrics import perturbation_budget_report


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a black-box SimBA attack on a sample image.")
    parser.add_argument("--image", type=str, default="sample.jpg", help="Image filename from examples/sample_images/")
    args = parser.parse_args()

    image_path = Path("examples/sample_images") / args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model("resnet18")
    clean_tensor = load_image(image_path)
    clean_tensor = clean_tensor.unsqueeze(0)

    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))
    logger.info("Target image: %s", image_path.name)
    logger.info("Clean prediction: class=%s confidence=%.4f", clean_label, clean_confidence)

    attack = SimBA(epsilon=0.02, max_queries=5000, early_stop=True)
    result = attack.generate(model, clean_tensor, clean_label)

    adversarial_tensor, queries_used = result
    perturbation = perturbation_budget_report(clean_tensor.squeeze(0), adversarial_tensor.squeeze(0), epsilon=attack.epsilon)
    adversarial_label, adversarial_confidence = predict(model, adversarial_tensor.squeeze(0))

    logger.info("Adversarial prediction: class=%s confidence=%.4f", adversarial_label, adversarial_confidence)
    logger.info("Changed: %s", clean_label != adversarial_label)
    logger.info("Queries used: %s", queries_used)
    logger.info("Perturbation L2: %.4f", perturbation["l2_norm"])
    logger.info("Perturbation L-infinity: %.4f", perturbation["linf_norm"])

    output_path = Path("examples/sample_images/adversarial_blackbox.png")
    tensor_to_image(adversarial_tensor.squeeze(0)).save(output_path)
    logger.info("Saved adversarial image to %s", output_path)

    if clean_label == adversarial_label and queries_used >= attack.max_queries:
        logger.warning("Attack failed within query budget — target is robust against this method.")

    confidence_drop_simba = clean_confidence - adversarial_confidence
    logger.info("")
    logger.info("Threat Model Summary")
    logger.info("-" * 60)
    logger.info("Method: SimBA")
    logger.info("Model access: None (query-only)")
    logger.info("Queries used: %s", queries_used)
    logger.info("Confidence drop: %.4f", confidence_drop_simba)
    logger.info("Perturbation L2: %.4f", perturbation["l2_norm"])
    logger.info("Perturbation L-infinity: %.4f", perturbation["linf_norm"])


if __name__ == "__main__":
    main()
