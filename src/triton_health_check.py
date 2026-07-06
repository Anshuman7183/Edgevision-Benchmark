"""Check NVIDIA Triton server and EdgeVision model readiness."""

from __future__ import annotations

import argparse
import sys

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


DEFAULT_TRITON_URL = "localhost:8000"
DEFAULT_MODEL_NAME = "edgevision_cnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Triton server liveness, server readiness, and model readiness."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_TRITON_URL,
        help=f"Triton HTTP endpoint host:port. Default: {DEFAULT_TRITON_URL}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Triton model name to check. Default: {DEFAULT_MODEL_NAME}",
    )
    return parser.parse_args()


def check_triton_health(url: str, model_name: str) -> bool:
    client = httpclient.InferenceServerClient(url=url)
    checks = [
        ("Server live", client.is_server_live),
        ("Server ready", client.is_server_ready),
        ("Model ready", lambda: client.is_model_ready(model_name)),
    ]

    all_ready = True
    for label, check in checks:
        try:
            ready = check()
        except InferenceServerException as exc:
            print(f"[FAIL] {label}: Triton returned an error: {exc}")
            all_ready = False
            continue
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")
            all_ready = False
            continue

        if ready:
            print(f"[OK] {label}")
        else:
            print(f"[FAIL] {label}: returned false")
            all_ready = False

    if all_ready:
        print(f"Success: Triton is ready and model '{model_name}' is ready at {url}.")
    else:
        print(f"Failure: Triton or model '{model_name}' is not ready at {url}.")
    return all_ready


def main() -> int:
    args = parse_args()
    return 0 if check_triton_health(url=args.url, model_name=args.model_name) else 1


if __name__ == "__main__":
    sys.exit(main())
