"""Interactive provider benchmark. Keys arrive via non-echoing stdin and stay in RAM.

First line: JSON mapping provider names to keys. Subsequent lines: commands.
No credential values or raw provider exception bodies are printed or saved.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENDPOINTS = {
    "OpenRouter": "https://openrouter.ai/api/v1",
    "NVIDIA": "https://integrate.api.nvidia.com/v1",
}


def read_keys():
    original = None
    if sys.stdin.isatty():
        import termios
        original = termios.tcgetattr(sys.stdin)
        settings = list(original)
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
    print("Ready for credentials on stdin; echo is disabled.", flush=True)
    try:
        return json.loads(sys.stdin.readline())
    finally:
        if original is not None:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, original)


def catalog(name, key):
    started = time.perf_counter()
    try:
        response = requests.get(ENDPOINTS[name] + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
        record = {"provider": name, "status": response.status_code, "seconds": round(time.perf_counter() - started, 2)}
        if response.ok:
            try:
                entries = response.json().get("data", [])
                record["models"] = [x["id"] for x in entries if isinstance(x, dict) and "id" in x]
            except (ValueError, AttributeError):
                record["error"] = "Response was not a model catalog"
        return record
    except Exception as exc:
        return {"provider": name, "error_type": type(exc).__name__}


def main():
    keys = read_keys()
    pipeline = None
    for line in sys.stdin:
        command = json.loads(line)
        if command["action"] == "quit":
            break
        if command["action"] == "checks":
            from dataclasses import replace
            from src.providers import ProviderConfig, OpenAICompatibleProvider
            from src.catalog import provider_error_message
            for name, model in command["targets"]:
                started = time.perf_counter()
                try:
                    cfg = replace(ProviderConfig.for_provider(name, model=model, api_key=keys.get(name, "")), request_timeout=8)
                    response = OpenAICompatibleProvider(cfg)._complete([{"role": "user", "content": "Reply with OK only."}], max_tokens=128)
                    print(json.dumps({"provider": name, "model": model, "success": bool(response.choices[0].message.content), "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                except Exception as exc:
                    record = {"provider": name, "model": model, "error": provider_error_message(exc, name), "seconds": round(time.perf_counter()-started, 2)}
                    # Classify access errors without logging raw bodies or credentials.
                    body = str(getattr(exc, "body", ""))
                    if not body and getattr(exc, "response", None) is not None:
                        body = exc.response.text
                    lowered = body.lower()
                    record["status"] = getattr(exc, "status_code", None) or (exc.response.status_code if getattr(exc, "response", None) is not None else None)
                    record["access_signals"] = [term for term in ("invalid", "expired", "only", "client", "permission", "disabled", "token", "region", "balance") if term in lowered]
                    print(json.dumps(record), flush=True)
            print("COMMAND COMPLETE", flush=True)
            continue
        if command["action"] == "browser":
            from browser_smoke import main as browser_main
            browser_main(keys["OpenRouter"])
            print("COMMAND COMPLETE", flush=True)
            continue
        if command["action"] == "catalogs":
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(catalog, name, key) for name, key in keys.items()]):
                    print(json.dumps(future.result()), flush=True)
        elif command["action"] == "ping":
            def ping(target):
                name, model = target
                started = time.perf_counter()
                try:
                    response = requests.post(ENDPOINTS[name] + "/chat/completions", headers={"Authorization": f"Bearer {keys[name]}"}, json={
                        "model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 32,
                    }, timeout=(10, 40))
                    record = {"provider": name, "model": model, "status": response.status_code, "seconds": round(time.perf_counter()-started, 2)}
                    if response.ok:
                        payload = response.json()
                        record["content_received"] = bool(payload.get("choices", [{}])[0].get("message", {}).get("content"))
                        record["actual_model"] = payload.get("model")
                    return record
                except Exception as exc:
                    return {"provider": name, "model": model, "error_type": type(exc).__name__, "seconds": round(time.perf_counter()-started, 2)}
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(ping, target) for target in command["targets"]]):
                    print(json.dumps(future.result()), flush=True)
        elif command["action"] == "benchmark":
            import importlib
            import src.providers
            importlib.reload(src.providers)
            from src.pipeline import DocumentRAGPipeline
            from src.providers import OpenAICompatibleProvider, ProviderConfig
            from src.catalog import discover_models, ModelOption
            if pipeline is None:
                pipeline = DocumentRAGPipeline(ROOT / "cache")
                files = ["training-compute-optimal-large-language-models.pdf", "efficient-streaming-language-models.pdf"]
                pipeline.ingest([(file, (ROOT / "cache/samples" / file).read_bytes()) for file in files])
                pipeline.retrieve("window size", document_ids={pipeline.documents[-1].document_id})
            selected = {pipeline.documents[-1].document_id}
            records = []
            for name, model in command["targets"]:
                options = discover_models(name)
                option = next((x for x in options if x.id == model), ModelOption(model, model))
                config = ProviderConfig(name, ENDPOINTS[name], model, keys[name], supports_images=bool(command.get("vision", option.vision)), context_length=32768, max_output_tokens=4096, disable_reasoning=option.disable_reasoning, json_output=option.json_output)
                provider = OpenAICompatibleProvider(config)
                for question in command.get("questions", ["What was the window size used for comparison and evaluation? Explain the settings for each model family and why they were chosen."]):
                    result = pipeline.answer(question, provider, selected, answer_mode=command.get("mode", "Detailed"), on_partial=lambda text: None)
                    record = asdict(result)
                    if command.get("debug_public_response"):
                        record["raw_public_response"] = getattr(provider, "last_raw_response", "")
                    records.append(record)
                    print(json.dumps({"provider": name, "requested_model": model, "model": result.model, "success": result.generation_succeeded, "words": len(result.answer.split()), "requests": result.request_count, "timings": result.timings, "warning": result.parse_warning, "answer": result.answer}), flush=True)
            output = ROOT / "evaluation_results" / (command.get("output", "hosted_benchmark") + ".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("COMMAND COMPLETE", flush=True)
    keys.clear()


if __name__ == "__main__":
    main()
