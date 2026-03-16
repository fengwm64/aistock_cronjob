#!/usr/bin/env python3
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

import requests

ANALYSIS_URL_TEMPLATE = "https://extapi.aistocklink.cn/api/cn/stocks/{symbol}/analysis"
PROFIT_FORECAST_URL_TEMPLATE = (
    "https://extapi.aistocklink.cn/api/cn/stock/{symbol}/profit-forecast"
)
KRONOS_PREDICT_BATCH_URL = "https://yingfeng64-kronos-api.hf.space/api/v1/predict/batch"
KRONOS_MAX_BATCH_SIZE = 20
RETRY_TIMES = 8
RETRY_BASE_DELAY_SECONDS = 1.0
T = TypeVar("T")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def retry_call(
    func: Callable[[], T], label: str, retries: int = RETRY_TIMES, base_delay: float = RETRY_BASE_DELAY_SECONDS
) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            delay = base_delay * (attempt + 1)
            print(
                f"[retry] {label} failed ({attempt + 1}/{retries + 1}): {exc}; sleep {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if last_exc is None:
        raise RuntimeError(f"{label} failed without exception")
    raise last_exc


def parse_watchlist_from_env(env_var: str) -> List[str]:
    raw = os.getenv(env_var, "")
    resolved_key = env_var
    if not str(raw).strip():
        for key in (env_var.upper(), env_var.lower()):
            value = os.getenv(key, "")
            if str(value).strip():
                raw = value
                resolved_key = key
                break

    if not str(raw).strip():
        # Case-insensitive fallback for environments that normalize key names.
        for key, value in os.environ.items():
            if key.lower() == env_var.lower() and str(value).strip():
                raw = value
                resolved_key = key
                break

    raw = str(raw).strip()
    if not raw:
        similar_keys = sorted(
            [k for k in os.environ.keys() if "WATCH" in k.upper() or "SYMBOL" in k.upper()]
        )[:20]
        raise RuntimeError(
            f"Environment variable {env_var} is empty. "
            f"Expected format: 601669;000617;603986. "
            f"Similar env keys in current process: {similar_keys}"
        )

    raw = raw.strip("\"'")
    # Support separators: ; ； , ， whitespace/newlines
    symbols = [part.strip() for part in re.split(r"[;；,，\s]+", raw) if part.strip()]
    if not symbols:
        raise RuntimeError(
            f"No valid symbols found in {resolved_key}. "
            f"Expected format: 601669;000617;603986"
        )

    # Keep order while removing duplicates
    deduped = list(dict.fromkeys(symbols))
    return deduped


def run_analysis_sse(symbol: str) -> Dict[str, Any]:
    def _request() -> Dict[str, Any]:
        url = ANALYSIS_URL_TEMPLATE.format(symbol=symbol)
        headers = {"Accept": "text/event-stream"}
        result: Dict[str, Any] = {
            "ok": False,
            "status_code": None,
            "event_count": 0,
            "done_received": False,
            "last_event": None,
        }

        with requests.post(url, headers=headers, stream=True, timeout=(15, 180)) as resp:
            result["status_code"] = resp.status_code
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        result["done_received"] = True
                        break
                    try:
                        event_obj = json.loads(payload)
                    except json.JSONDecodeError:
                        event_obj = payload
                    result["last_event"] = event_obj
                    result["event_count"] = int(result["event_count"]) + 1

            result["ok"] = True
        return result

    return retry_call(_request, label=f"analysis_sse:{symbol}")


def fetch_profit_forecast(symbol: str) -> Dict[str, Any]:
    def _request() -> Dict[str, Any]:
        url = PROFIT_FORECAST_URL_TEMPLATE.format(symbol=symbol)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return {"ok": True, "status_code": resp.status_code, "data": resp.json()}

    return retry_call(_request, label=f"profit_forecast:{symbol}")


def chunked_symbols(symbols: List[str], chunk_size: int) -> List[List[str]]:
    return [symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)]


def submit_predict_batch(symbols: List[str]) -> Dict[str, Any]:
    def _request() -> Dict[str, Any]:
        payload = {
            "requests": [
                {
                    "symbol": symbol,
                    "lookback": 256,
                    "pred_len": 5,
                    "sample_count": 30,
                    "mode": "simple",
                    "include_volume": False,
                }
                for symbol in symbols
            ]
        }
        resp = requests.post(KRONOS_PREDICT_BATCH_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("predict batch submit response is not a JSON object")

        batch_id = str(data.get("batch_id", "")).strip()
        task_ids = data.get("task_ids")
        if not batch_id:
            raise RuntimeError("predict batch submit missing batch_id")
        if not isinstance(task_ids, list) or len(task_ids) != len(symbols):
            raise RuntimeError(
                "predict batch submit returned invalid task_ids length: "
                f"expect {len(symbols)}, got {len(task_ids) if isinstance(task_ids, list) else 'N/A'}"
            )

        return {
            "ok": True,
            "status_code": resp.status_code,
            "data": data,
            "batch_id": batch_id,
            "task_ids": [str(task_id) for task_id in task_ids],
        }

    return retry_call(
        _request,
        label=f"predict_batch_submit:{','.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}",
    )


def poll_predict_batch(batch_id: str, poll_interval: float, poll_timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + poll_timeout
    poll_url = f"{KRONOS_PREDICT_BATCH_URL}/{batch_id}"
    last_data: Dict[str, Any] = {}

    while True:
        if time.monotonic() > deadline:
            return {
                "ok": False,
                "batch_id": batch_id,
                "status": "timeout",
                "data": last_data,
            }

        def _request() -> Dict[str, Any]:
            resp = requests.get(poll_url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"raw": data}
            return data

        try:
            data = retry_call(_request, label=f"predict_batch_poll:{batch_id}")
        except Exception as exc:
            return {
                "ok": False,
                "batch_id": batch_id,
                "status": "poll_error",
                "error": str(exc),
                "data": last_data,
            }

        last_data = data if isinstance(data, dict) else {"raw": data}
        status = str(last_data.get("status", "")).lower()

        if status in {"done", "partial", "failed"}:
            return {
                "ok": status in {"done", "partial"},
                "batch_id": batch_id,
                "status": status,
                "data": last_data,
            }

        time.sleep(poll_interval)


def process_symbol(symbol: str) -> Dict[str, Any]:
    started_at = now_str()
    tid = threading.get_ident()
    print(f"[{started_at}] [thread={tid}] start symbol={symbol}", flush=True)
    output: Dict[str, Any] = {"symbol": symbol, "started_at": started_at}

    try:
        output["profit_forecast"] = fetch_profit_forecast(symbol)
    except Exception as exc:
        output["profit_forecast"] = {"ok": False, "error": str(exc)}

    try:
        output["analysis"] = run_analysis_sse(symbol)
    except Exception as exc:
        output["analysis"] = {"ok": False, "error": str(exc)}

    output["finished_at"] = now_str()
    print(
        f"[{output['finished_at']}] [thread={tid}] done symbol={symbol}",
        flush=True,
    )
    return output


def run_predict_batches(
    symbols: List[str], poll_interval: float, poll_timeout: float, max_workers: int
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {
        symbol: {
            "predict_submit": {"ok": False, "error": "not_started"},
            "predict_poll": {"ok": False, "error": "not_started"},
        }
        for symbol in symbols
    }
    batch_jobs: List[Dict[str, Any]] = []

    for batch_symbols in chunked_symbols(symbols, KRONOS_MAX_BATCH_SIZE):
        try:
            submit_result = submit_predict_batch(batch_symbols)
            batch_id = submit_result["batch_id"]
            task_ids = submit_result["task_ids"]
            symbol_task_pairs = list(zip(batch_symbols, task_ids))
            batch_jobs.append({"batch_id": batch_id, "symbol_task_pairs": symbol_task_pairs})

            for symbol, task_id in symbol_task_pairs:
                results[symbol]["predict_submit"] = {
                    "ok": True,
                    "status_code": submit_result["status_code"],
                    "batch_id": batch_id,
                    "task_id": task_id,
                }
        except Exception as exc:
            err = str(exc)
            for symbol in batch_symbols:
                results[symbol]["predict_submit"] = {"ok": False, "error": err}
                results[symbol]["predict_poll"] = {"ok": False, "error": err}

    if not batch_jobs:
        return results

    poll_workers = max(2, min(max_workers, len(batch_jobs)))
    with ThreadPoolExecutor(max_workers=poll_workers) as executor:
        futures = {
            executor.submit(
                poll_predict_batch, job["batch_id"], poll_interval, poll_timeout
            ): job
            for job in batch_jobs
        }
        for fut in as_completed(futures):
            job = futures[fut]
            batch_id = job["batch_id"]
            symbol_task_pairs = job["symbol_task_pairs"]
            try:
                poll_result = fut.result()
            except Exception as exc:
                err = str(exc)
                for symbol, task_id in symbol_task_pairs:
                    results[symbol]["predict_poll"] = {
                        "ok": False,
                        "batch_id": batch_id,
                        "task_id": task_id,
                        "error": err,
                    }
                continue

            batch_status = str(poll_result.get("status", "")).lower()
            batch_data = poll_result.get("data", {})
            tasks = batch_data.get("tasks", []) if isinstance(batch_data, dict) else []
            task_map: Dict[str, Dict[str, Any]] = {}
            if isinstance(tasks, list):
                for task in tasks:
                    if isinstance(task, dict) and task.get("task_id"):
                        task_map[str(task["task_id"])] = task

            for symbol, task_id in symbol_task_pairs:
                task_data = task_map.get(task_id)
                if task_data is None:
                    results[symbol]["predict_poll"] = {
                        "ok": False,
                        "batch_id": batch_id,
                        "task_id": task_id,
                        "batch_status": batch_status,
                        "status": "missing",
                        "error": "task result not found in batch response",
                    }
                    continue

                task_status = str(task_data.get("status", "")).lower()
                results[symbol]["predict_poll"] = {
                    "ok": task_status == "done",
                    "batch_id": batch_id,
                    "task_id": task_id,
                    "batch_status": batch_status,
                    "status": task_status,
                    "data": task_data,
                }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task: watchlist from env + analysis/profit + kronos batch predict (multi-thread)."
    )
    parser.add_argument(
        "--env-var",
        type=str,
        default="WATCHLIST_SYMBOLS",
        help="Environment variable containing semicolon-separated symbols.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent symbols (default: 8, minimum effective value: 2).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval seconds for kronos predict task.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=3600.0,
        help="Polling timeout seconds for each kronos predict task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_str()
    worker_count = max(2, args.max_workers)
    symbols = parse_watchlist_from_env(args.env_var)

    symbol_outputs: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_symbol, symbol): symbol
            for symbol in symbols
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                symbol_outputs[symbol] = fut.result()
            except Exception as exc:
                symbol_outputs[symbol] = {
                    "symbol": symbol,
                    "started_at": None,
                    "finished_at": now_str(),
                    "fatal_error": str(exc),
                }

    predict_outputs = run_predict_batches(
        symbols, args.poll_interval, args.poll_timeout, worker_count
    )
    results: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol_result = symbol_outputs.get(
            symbol,
            {
                "symbol": symbol,
                "started_at": None,
                "finished_at": now_str(),
                "fatal_error": "symbol result missing",
            },
        )
        symbol_result.update(
            predict_outputs.get(
                symbol,
                {
                    "predict_submit": {"ok": False, "error": "predict result missing"},
                    "predict_poll": {"ok": False, "error": "predict result missing"},
                },
            )
        )
        results.append(symbol_result)

    finished_at = now_str()
    analysis_ok = sum(1 for item in results if item.get("analysis", {}).get("ok") is True)
    profit_ok = sum(
        1 for item in results if item.get("profit_forecast", {}).get("ok") is True
    )
    predict_ok = sum(1 for item in results if item.get("predict_poll", {}).get("ok") is True)
    print(
        f"[watchlist_task] finished_at={finished_at} total={len(results)} "
        f"analysis_ok={analysis_ok} profit_ok={profit_ok} predict_ok={predict_ok}"
    )


if __name__ == "__main__":
    main()
