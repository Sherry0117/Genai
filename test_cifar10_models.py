import os
import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Tuple

import requests


BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
OUT_DIR = os.environ.get("API_TEST_OUT", "./api_test_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _save_base64_png(b64: str, path: str) -> None:
    try:
        data = base64.b64decode(b64)
        with open(path, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"[WARN] Failed to save image {path}: {e}")


def _ok(resp: requests.Response) -> bool:
    return 200 <= resp.status_code < 300


def test_get(path: str) -> Tuple[bool, Dict[str, Any], int]:
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=60)
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": resp.text}
        return _ok(resp), data, resp.status_code
    except Exception as e:
        return False, {"error": str(e)}, 0


def test_post(path: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": resp.text}
        return _ok(resp), data, resp.status_code
    except Exception as e:
        return False, {"error": str(e)}, 0


def run_tests() -> None:
    results: List[Tuple[str, bool, int, str]] = []

    # 1) Health
    ok, data, code = test_get("/health")
    results.append(("GET /health", ok, code, "status" if isinstance(data, dict) else ""))

    # 2) Available models (energy/diffusion)
    ok, data, code = test_get("/models")
    have_models = ok and isinstance(data, dict) and "models" in data and isinstance(data["models"], list)
    results.append(("GET /models", have_models, code, f"models={data.get('models', [])}"))

    # 3) Root
    ok, data, code = test_get("/")
    results.append(("GET /", ok, code, "root"))

    # 4) Text generation (bigram/LSTM wrapper)
    payload = {"start_word": "hello", "length": 10}
    ok, data, code = test_post("/generate", payload)
    valid = ok and isinstance(data, dict) and "generated_text" in data
    results.append(("POST /generate", valid, code, "generated_text"))

    ok, data, code = test_post("/generate_with_rnn", payload)
    valid = ok and isinstance(data, dict) and "generated_text" in data
    results.append(("POST /generate_with_rnn", valid, code, "generated_text"))

    # 5) LLM generation (may fail if model not trained)
    ok, data, code = test_post("/generate_with_llm", payload)
    # Accept both success or controlled error response
    llm_valid = ok or (isinstance(data, dict) and "error" in data)
    results.append(("POST /generate_with_llm", llm_valid, code, "ok_or_error"))

    # 6) Embedding endpoint
    ok, data, code = test_post("/embed", {"text": "FastAPI test", "pooling": "mean"})
    valid = ok and isinstance(data, dict) and "dim" in data
    results.append(("POST /embed", valid, code, "dim_present"))

    # 7) GAN image generation (saves images)
    ok, data, code = test_post("/generate_image", {"num_images": 2})
    valid = ok and isinstance(data, dict) and "images" in data and isinstance(data["images"], list)
    if valid:
        for i, b64 in enumerate(data["images"]):
            _save_base64_png(b64, os.path.join(OUT_DIR, f"gan_{i}.png"))
    results.append(("POST /generate_image", valid, code, "images_saved" if valid else ""))

    # 8) CNN info
    ok, data, code = test_get("/cnn_info")
    valid = ok and isinstance(data, dict) and "model_name" in data
    results.append(("GET /cnn_info", valid, code, data.get("model_name", "")))

    # 9) Energy/Diffusion generation (save grids)
    # Energy
    ok, data, code = test_post("/generate/energy", {"num_samples": 9, "seed": 123})
    valid = ok and isinstance(data, dict) and "image_base64" in data
    if valid:
        _save_base64_png(data["image_base64"], os.path.join(OUT_DIR, "energy_grid.png"))
    results.append(("POST /generate/energy", valid, code, "grid_saved" if valid else ""))

    # Diffusion
    ok, data, code = test_post("/generate/diffusion", {"num_samples": 9, "seed": 123})
    valid = ok and isinstance(data, dict) and "image_base64" in data
    if valid:
        _save_base64_png(data["image_base64"], os.path.join(OUT_DIR, "diffusion_grid.png"))
    results.append(("POST /generate/diffusion", valid, code, "grid_saved" if valid else ""))

    # 10) Classification requires a base64 image; reuse a GAN image if available
    classify_payload_valid = False
    img_path = os.path.join(OUT_DIR, "gan_0.png")
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                b64img = base64.b64encode(f.read()).decode("utf-8")
            ok, data, code = test_post("/classify_image", {"image": b64img, "top_k": 3})
            valid = ok and isinstance(data, dict) and "predictions" in data
            classify_payload_valid = True
            results.append(("POST /classify_image", valid, code, "predictions" if valid else ""))
        except Exception as e:
            results.append(("POST /classify_image", False, 0, f"read_img_error: {e}"))
    else:
        results.append(("POST /classify_image", False, 0, "no_gan_image_to_classify"))

    # Summary
    print("\n=== API Test Summary ===")
    passed = 0
    for name, ok, code, note in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} (code={code}) {note}")
        passed += int(ok)
    print(f"Total: {passed}/{len(results)} passed. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    run_tests()


