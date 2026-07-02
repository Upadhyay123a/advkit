from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from advkit.attacks.fgsm import FGSM
from advkit.attacks.pgd import PGD
from advkit.attacks.blackbox import SimBA
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image, tensor_to_image
from advkit.utils.metrics import perturbation_budget_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Robust path (works no matter where CLI is run from)
#SAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "sample_images"
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample_images"


# Registry mapping CLI method names -> attack config
ATTACK_REGISTRY: dict[str, dict[str, Any]] = {
    "fgsm": {
        "class": FGSM,
        "description": "Fast Gradient Sign Method — single-step white-box attack.",
        "access": "White-box (gradients)",
    },
    "pgd": {
        "class": PGD,
        "description": "Projected Gradient Descent — iterative white-box attack.",
        "access": "White-box (gradients)",
    },
    "blackbox": {
        "class": SimBA,
        "description": "SimBA — query-only black-box attack.",
        "access": "Black-box (query-only)",
    },
}


def build_attack(method: str, args: argparse.Namespace):
    entry = ATTACK_REGISTRY[method]
    attack_cls = entry["class"]

    if method == "fgsm":
        return attack_cls(epsilon=args.epsilon)

    if method == "pgd":
        return attack_cls(
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=not args.no_random_start,
            early_stop=args.early_stop,
        )

    if method == "blackbox":
        return attack_cls(
            epsilon=args.epsilon,
            max_queries=args.max_queries,
            early_stop=args.early_stop,
        )

    raise ValueError(f"Unknown method: {method}")


def run_attack(args: argparse.Namespace) -> int:
    image_path = SAMPLE_DIR / args.image

    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    logger.info("=" * 60)
    logger.info("AdvKit — Adversarial Attack Toolkit")
    logger.info("=" * 60)

    model = load_model(args.model)

    clean_tensor = load_image(image_path).unsqueeze(0)
    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))

    logger.info("Clean prediction: %s (%.4f)", clean_label, clean_confidence)

    attack = build_attack(args.method, args)
    result = attack.generate(model, clean_tensor, clean_label)

    if isinstance(result, tuple):
        adversarial_tensor, queries_used = result
    else:
        adversarial_tensor, queries_used = result, None

    adv_label, adv_conf = predict(model, adversarial_tensor.squeeze(0))

    changed = clean_label != adv_label

    perturbation = perturbation_budget_report(
        clean_tensor.squeeze(0),
        adversarial_tensor.squeeze(0),
        epsilon=args.epsilon,
    )

    logger.info("-" * 60)
    logger.info("Adversarial prediction: %s (%.4f)", adv_label, adv_conf)
    logger.info("Attack success: %s", changed)

    if queries_used is not None:
        logger.info("Queries used: %s", queries_used)

    logger.info("Confidence drop: %.4f", clean_confidence - adv_conf)
    logger.info("L2 norm: %.4f", perturbation["l2_norm"])
    logger.info("L∞ norm: %.4f", perturbation["linf_norm"])

    # Save output
    output_path = (
        SAMPLE_DIR / args.output
        if args.output
        else SAMPLE_DIR / f"adversarial_{args.method}.png"
    )

    tensor_to_image(adversarial_tensor.squeeze(0)).save(output_path)
    logger.info("Saved to: %s", output_path)

    return 0


def list_methods(_: argparse.Namespace) -> int:
    print("\nAvailable attack methods:\n")
    for name, entry in ATTACK_REGISTRY.items():
        print(f"{name:10s} [{entry['access']}]")
        print(f"  - {entry['description']}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advkit",
        description="AdvKit — Adversarial Attack Toolkit",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # attack command
    attack_parser = subparsers.add_parser("attack")
    attack_parser.add_argument("--image", required=True)
    attack_parser.add_argument("--model", default="resnet18")
    attack_parser.add_argument("--method", required=True, choices=list(ATTACK_REGISTRY.keys()))
    attack_parser.add_argument("--epsilon", type=float, default=0.02)
    attack_parser.add_argument("--alpha", type=float, default=0.005)
    attack_parser.add_argument("--num-steps", type=int, default=10)
    attack_parser.add_argument("--max-queries", type=int, default=5000)

    attack_parser.add_argument("--no-random-start", action="store_true")

    # FIXED: proper toggle
    attack_parser.add_argument("--early-stop", action="store_true")
    attack_parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    attack_parser.set_defaults(early_stop=True)

    attack_parser.add_argument("--output", default=None)

    attack_parser.set_defaults(func=run_attack)

    # list methods
    list_parser = subparsers.add_parser("list-methods")
    list_parser.set_defaults(func=list_methods)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()