#!/usr/bin/env python3
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

STOCK_RANK_URL = "https://extapi.aistocklink.cn/api/cn/market/stockrank/"
ANALYSIS_URL_TEMPLATE = "https://extapi.aistocklink.cn/api/cn/stocks/{symbol}/analysis"
PROFIT_FORECAST_URL_TEMPLATE = (
    "https://extapi.aistocklink.cn/api/cn/stock/{symbol}/profit-forecast"
)
KRONOS_PREDICT_URL = "https://yingfeng64-kronos-api.hf.space/api/v1/predict"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_stock_rank() -> Dict[str, Any]:
    resp = requests.get(STOCK_RANK_URL, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_symbols(stock_rank_resp: Dict[str, Any]) -> List[str]:
    rank_list = stock_rank_resp.get("data", {}).get("人气榜", [])
    symbols: List[str] = []
    for item in rank_list:
        symbol = str(item.get("股票代码", "")).strip()
        if symbol:
            symbols.append(symbol)
    return symbols


def run_analysis_sse(symbol: str) -> Dict[str, Any]:
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


def fetch_profit_forecast(symbol: str) -> Dict[str, Any]:
    url = PROFIT_FORECAST_URL_TEMPLATE.format(symbol=symbol)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return {"ok": True, "status_code": resp.status_code, "data": resp.json()}


def submit_predict(symbol: str) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "lookback": 256,
        "pred_len": 5,
        "sample_count": 30,
        "mode": "simple",
        "include_volume": False,
    }
    resp = requests.post(KRONOS_PREDICT_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {"ok": True, "status_code": resp.status_code, "data": data}


def extract_task_id(submit_resp: Dict[str, Any]) -> Optional[str]:
    data = submit_resp.get("data", {})
    if not isinstance(data, dict):
        return None
    for key in ("task_id", "id", "taskId"):
        value = data.get(key)
        if value:
            return str(value)
    return None


def poll_predict(task_id: str, poll_interval: float, poll_timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + poll_timeout
    poll_url = f"{KRONOS_PREDICT_URL}/{task_id}"
    last_data: Dict[str, Any] = {}

    while True:
        if time.monotonic() > deadline:
            return {
                "ok": False,
                "task_id": task_id,
                "status": "timeout",
                "data": last_data,
            }

        resp = requests.get(poll_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        last_data = data if isinstance(data, dict) else {"raw": data}
        status = str(last_data.get("status", "")).lower()

        if status == "done":
            return {"ok": True, "task_id": task_id, "status": status, "data": last_data}
        if status in {"failed", "error", "cancelled", "canceled"}:
            return {"ok": False, "task_id": task_id, "status": status, "data": last_data}

        time.sleep(poll_interval)


def process_symbol(symbol: str, poll_interval: float, poll_timeout: float) -> Dict[str, Any]:
    started_at = now_str()
    tid = threading.get_ident()
    print(f"[{started_at}] [thread={tid}] start symbol={symbol}", flush=True)
    output: Dict[str, Any] = {"symbol": symbol, "started_at": started_at}

    try:
        output["analysis"] = run_analysis_sse(symbol)
    except Exception as exc:
        output["analysis"] = {"ok": False, "error": str(exc)}

    try:
        output["profit_forecast"] = fetch_profit_forecast(symbol)
    except Exception as exc:
        output["profit_forecast"] = {"ok": False, "error": str(exc)}

    try:
        submit_data = submit_predict(symbol)
        output["predict_submit"] = submit_data
        task_id = extract_task_id(submit_data)
        if task_id:
            output["predict_poll"] = poll_predict(task_id, poll_interval, poll_timeout)
        else:
            output["predict_poll"] = {"ok": False, "error": "task_id not found"}
    except Exception as exc:
        output["predict_submit"] = {"ok": False, "error": str(exc)}
        output["predict_poll"] = {"ok": False, "error": str(exc)}

    output["finished_at"] = now_str()
    print(
        f"[{output['finished_at']}] [thread={tid}] done symbol={symbol}",
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task1: stock rank + per-symbol analysis/profit/predict (multi-thread)."
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
        default=300.0,
        help="Polling timeout seconds for each kronos predict task.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"task1_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_str()
    worker_count = max(2, args.max_workers)

    rank_resp = fetch_stock_rank()
    symbols = extract_symbols(rank_resp)
    if not symbols:
        raise RuntimeError("No symbols extracted from stock rank response.")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_symbol, symbol, args.poll_interval, args.poll_timeout
            ): symbol
            for symbol in symbols
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "started_at": None,
                        "finished_at": now_str(),
                        "fatal_error": str(exc),
                    }
                )

    output = {
        "task": "task1",
        "started_at": started,
        "finished_at": now_str(),
        "stock_rank": rank_resp,
        "symbols": symbols,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
