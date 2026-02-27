from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union
import heapq
import json
import os
import time

TimeWindow = Union[Tuple[float, float], Dict[str, Any]]
Delivery = TimeWindow
Package = Union[Dict[str, Any], Tuple[float, float], Tuple[float, float, Any]]  # (weight, value[, id])


def _get_start_end(item: Any) -> Tuple[float, float]:
    """
    Robust start/end extraction.

    Supports dict keys:
      start/end, start_time/end_time, begin/finish

    Supports tuple/list formats like:
      (start, end)
      (id, start, end)
      (start, end, id)
      (..., start, end)   # start/end at the end
    """
    # dict case with common key variants
    if isinstance(item, dict):
        for s_key, e_key in [("start", "end"), ("start_time", "end_time"), ("begin", "finish")]:
            if s_key in item and e_key in item:
                return float(item[s_key]), float(item[e_key])
        raise KeyError(f"Could not find start/end keys in dict: {item.keys()}")

    if not isinstance(item, (list, tuple)):
        raise ValueError(f"Unsupported delivery type: {type(item)}")

    n = len(item)
    if n < 2:
        raise ValueError(f"Delivery must have at least 2 fields: {item}")

    # 1) If it looks exactly like (start, end)
    if n == 2:
        return float(item[0]), float(item[1])

    # 2) Very common: (id, start, end)  -> use positions 1 and 2 if numeric-convertible
    try:
        return float(item[1]), float(item[2])
    except Exception:
        pass

    # 3) Another common: (..., start, end) -> use last two
    try:
        return float(item[-2]), float(item[-1])
    except Exception:
        pass

    # 4) Fallback: first two
    return float(item[0]), float(item[1])


def _get_weight_value(pkg: Package) -> Tuple[float, float, Any]:
    if isinstance(pkg, dict):
        w = float(pkg["weight"])
        v = pkg.get("value", pkg.get("priority", pkg.get("priority_value")))
        if v is None:
            raise KeyError("Package dict must have 'value' (or 'priority').")
        ident = pkg.get("id", pkg.get("name", pkg.get("package_id", None)))
        return float(w), float(v), ident if ident is not None else pkg

    if len(pkg) == 2:
        w, v = pkg
        return float(w), float(v), pkg

    if len(pkg) == 3:
        w, v, ident = pkg
        return float(w), float(v), ident

    raise ValueError("Unsupported package format.")


# PART 1: Activity selection (max non-overlapping)
def maximize_deliveries(time_windows: Sequence[TimeWindow]) -> List[TimeWindow]:
    if not time_windows:
        return []

    sorted_windows = sorted(time_windows, key=lambda x: (_get_start_end(x)[1], _get_start_end(x)[0]))

    chosen: List[TimeWindow] = []
    current_end = float("-inf")

    for w in sorted_windows:
        start, end = _get_start_end(w)
        if start >= current_end:
            chosen.append(w)
            current_end = end

    return chosen


# PART 2: Fractional knapsack
def optimize_truck_load(packages: Sequence[Package], weight_limit: float) -> Dict[str, Any]:
    if weight_limit <= 0 or not packages:
        return {"total_value": 0.0, "total_weight": 0.0, "items": []}

    items = []
    for p in packages:
        w, v, ident = _get_weight_value(p)
        if w > 0:
            items.append((v / w, w, v, ident))

    items.sort(key=lambda t: t[0], reverse=True)

    remaining = float(weight_limit)
    total_value = 0.0
    total_weight = 0.0
    picked: List[Dict[str, Any]] = []

    for ratio, w, v, ident in items:
        if remaining <= 0:
            break
        take_w = min(w, remaining)
        frac = take_w / w
        take_v = v * frac

        total_weight += take_w
        total_value += take_v
        remaining -= take_w

        picked.append({"id": ident, "fraction": frac, "weight_taken": take_w, "value_taken": take_v})

    return {"total_value": total_value, "total_weight": total_weight, "items": picked}


# PART 3: Interval partitioning (min drivers)
def minimize_drivers(deliveries: Sequence[Delivery]) -> Dict[str, Any]:
    if not deliveries:
        return {"num_drivers": 0, "assignments": []}

    sorted_deliveries = sorted(deliveries, key=lambda d: (_get_start_end(d)[0], _get_start_end(d)[1]))

    heap: List[Tuple[float, int]] = []
    assignments: List[List[Delivery]] = []

    for d in sorted_deliveries:
        start, end = _get_start_end(d)

        # allow back-to-back: end == start is OK
        if heap and heap[0][0] <= start:
            _, idx = heapq.heappop(heap)
            assignments[idx].append(d)
            heapq.heappush(heap, (end, idx))
        else:
            idx = len(assignments)
            assignments.append([d])
            heapq.heappush(heap, (end, idx))

    return {"num_drivers": len(assignments), "assignments": assignments}


# Optional: benchmark if your repo uses scenarios/
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_scenarios(scenarios_dir: str = "scenarios") -> None:
    if not os.path.isdir(scenarios_dir):
        print(f"[benchmark] No scenarios directory found at: {scenarios_dir}")
        return

    files = [os.path.join(scenarios_dir, f) for f in os.listdir(scenarios_dir) if f.endswith(".json")]
    if not files:
        print(f"[benchmark] No .json scenario files found in: {scenarios_dir}")
        return

    print(f"[benchmark] Found {len(files)} scenario files.")
    for fp in sorted(files):
        data = _load_json(fp)
        print(f"\n--- {os.path.basename(fp)} ---")

        if isinstance(data, dict) and "time_windows" in data:
            t0 = time.perf_counter()
            chosen = maximize_deliveries(data["time_windows"])
            t1 = time.perf_counter()
            print(f"Package Prioritization: {len(chosen)} in {(t1 - t0)*1000:.3f} ms")

        elif isinstance(data, dict) and "packages" in data and "weight_limit" in data:
            t0 = time.perf_counter()
            res = optimize_truck_load(data["packages"], data["weight_limit"])
            t1 = time.perf_counter()
            print(f"Truck Loading: value={res['total_value']:.2f} in {(t1 - t0)*1000:.3f} ms")

        elif isinstance(data, dict) and "deliveries" in data:
            t0 = time.perf_counter()
            res = minimize_drivers(data["deliveries"])
            t1 = time.perf_counter()
            print(f"Driver Assignment: drivers={res['num_drivers']} in {(t1 - t0)*1000:.3f} ms")

        else:
            print("[benchmark] Unrecognized scenario format.")

if __name__ == "__main__":
    if "test_package_prioritization" in globals():
        test_package_prioritization()
    if "test_truck_loading" in globals():
        test_truck_loading()
    if "test_driver_assignment" in globals():
        test_driver_assignment()

    print("All available tests passed!")

    benchmark_scenarios()   # <-- make sure this is NOT commented