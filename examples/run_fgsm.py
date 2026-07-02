from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advkit.attacks.fgsm import FGSM
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image, tensor_to_image


def main() -> None:
    image_path = Path("examples/sample_images/sample.jpg")
    model = load_model("resnet18")

    clean_tensor = load_image(image_path)
    clean_tensor = clean_tensor.unsqueeze(0)

    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))
    print(f"Before: class={clean_label} confidence={clean_confidence:.4f}")

    attack = FGSM(epsilon=0.03)
    adversarial_tensor = attack.generate(model, clean_tensor, clean_label)
    adversarial_label, adversarial_confidence = predict(model, adversarial_tensor.squeeze(0))

    print(f"After : class={adversarial_label} confidence={adversarial_confidence:.4f}")
    print(f"Changed: {clean_label != adversarial_label}")

    output_path = Path("examples/sample_images/adversarial.png")
    tensor_to_image(adversarial_tensor.squeeze(0)).save(output_path)
    print(f"Saved adversarial image to {output_path}")


if __name__ == "__main__":
    main()
