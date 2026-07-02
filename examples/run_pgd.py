from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from advkit.attacks.pgd import PGD
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image, tensor_to_image


def generate_with_tracking(attack, model, image_tensor, true_label):
    """Run PGD attack and track when the prediction first changes."""
    original = image_tensor.clone().detach()
    adversarial = original.clone().detach()
    flip_step = None

    for step in range(attack.num_steps):
        adversarial.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(adversarial)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([true_label], dtype=torch.long))
            loss.backward()
            gradient = adversarial.grad.detach()

        adversarial = adversarial.detach() + attack.alpha * gradient.sign()
        adversarial = torch.clamp(adversarial, original - attack.epsilon, original + attack.epsilon)
        adversarial = torch.clamp(adversarial, 0.0, 1.0)

        # Check if prediction changed at this step
        if flip_step is None:
            current_pred, _ = predict(model, adversarial)
            if current_pred != true_label:
                flip_step = step + 1

    return adversarial.detach(), flip_step


def main() -> None:
    image_path = Path("examples/sample_images/sample.jpg")
    model = load_model("resnet18")

    clean_tensor = load_image(image_path)
    clean_tensor = clean_tensor.unsqueeze(0)

    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))
    print(f"Before: class={clean_label} confidence={clean_confidence:.4f}")

    attack = PGD(epsilon=0.03, alpha=0.005, num_steps=10)
    adversarial_tensor, flip_step = generate_with_tracking(attack, model, clean_tensor, clean_label)
    adversarial_label, adversarial_confidence = predict(model, adversarial_tensor.squeeze(0))

    print(f"After : class={adversarial_label} confidence={adversarial_confidence:.4f}")
    print(f"Changed: {clean_label != adversarial_label}")
    if flip_step is not None:
        print(f"Prediction flipped at step: {flip_step}/{attack.num_steps}")
    else:
        print(f"Prediction did not flip within {attack.num_steps} steps")

    output_path = Path("examples/sample_images/adversarial_pgd.jpg")
    tensor_to_image(adversarial_tensor.squeeze(0)).save(output_path)
    print(f"Saved adversarial image to {output_path}")


if __name__ == "__main__":
    main()
