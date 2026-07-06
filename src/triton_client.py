"""Run one CIFAR-10 sample through the Triton-served EdgeVision model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from data_loader import CIFAR10_CLASSES, get_cifar10_datasets


DEFAULT_TRITON_URL = "localhost:8000"
DEFAULT_MODEL_NAME = "edgevision_cnn"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one preprocessed CIFAR-10 test sample to Triton over HTTP."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_TRITON_URL,
        help=f"Triton HTTP endpoint host:port. Default: {DEFAULT_TRITON_URL}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Triton model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="CIFAR-10 test sample index to send. Default: 0",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the CIFAR-10 dataset. Default: data",
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def load_sample(data_dir: Path, sample_index: int) -> tuple[np.ndarray, int]:
    _, test_dataset = get_cifar10_datasets(data_dir=data_dir, download=False)
    if sample_index < 0 or sample_index >= len(test_dataset):
        raise IndexError(
            f"sample-index {sample_index} is outside the CIFAR-10 test range "
            f"0 to {len(test_dataset) - 1}."
        )

    image, label = test_dataset[sample_index]
    batch = image.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)
    if batch.shape != (1, 3, 32, 32):
        raise ValueError(f"Expected input shape (1, 3, 32, 32), got {batch.shape}.")
    return batch, int(label)


def infer(
    url: str,
    model_name: str,
    input_batch: np.ndarray,
) -> np.ndarray:
    client = httpclient.InferenceServerClient(url=url)
    infer_input = httpclient.InferInput(INPUT_NAME, input_batch.shape, "FP32")
    infer_input.set_data_from_numpy(input_batch)
    requested_output = httpclient.InferRequestedOutput(OUTPUT_NAME)

    response = client.infer(
        model_name=model_name,
        inputs=[infer_input],
        outputs=[requested_output],
    )
    logits = response.as_numpy(OUTPUT_NAME)
    if logits is None:
        raise RuntimeError(f"Triton response did not include output '{OUTPUT_NAME}'.")
    return logits


def main() -> int:
    args = parse_args()

    try:
        input_batch, true_class_index = load_sample(
            data_dir=args.data_dir,
            sample_index=args.sample_index,
        )
        logits = infer(
            url=args.url,
            model_name=args.model_name,
            input_batch=input_batch,
        )
    except InferenceServerException as exc:
        print(f"Triton inference failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Inference client failed: {exc}")
        return 1

    probabilities = softmax(logits)
    predicted_class_index = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0, predicted_class_index])
    correct = predicted_class_index == true_class_index

    print(f"Sample index: {args.sample_index}")
    print(f"Predicted class: {CIFAR10_CLASSES[predicted_class_index]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"True class: {CIFAR10_CLASSES[true_class_index]}")
    print(f"Correct: {correct}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
