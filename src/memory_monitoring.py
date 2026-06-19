import os
import sys
import threading
import time

def _get_system_memory_usage_gb():
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        meminfo_kb = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) < 2:
                continue
            key = parts[0]
            value_tokens = parts[1].strip().split()
            if not value_tokens:
                continue
            # Values are reported in kB
            meminfo_kb[key] = int(value_tokens[0])

        total_kb = meminfo_kb.get("MemTotal")
        avail_kb = meminfo_kb.get("MemAvailable")
        if total_kb is None or avail_kb is None:
            return None
        used_kb = total_kb - avail_kb
        used_gb = used_kb / (1024 * 1024)
        total_gb = total_kb / (1024 * 1024)
        percent = (used_kb / total_kb) * 100.0
        return used_gb, total_gb, percent
    except Exception:
        return None

def _memory_logger_loop(interval_seconds: int = 10, kill_threshold_percent: float = 95.0):
    while True:
        usage = _get_system_memory_usage_gb()
        if usage is not None:
            used_gb, total_gb, percent = usage
            print(f"[mem] {used_gb:5.2f}/{total_gb:5.2f} GB ({percent:5.1f}%)", flush=True)
            if kill_threshold_percent is not None and percent >= kill_threshold_percent:
                # Hard-kill the whole process to avoid an OOM thrash/hang.  We run in
                # a daemon thread, so a plain raise would only kill this thread; and
                # under memory pressure the main thread is usually deep in a C-level
                # numpy/polars call holding the GIL, so signals won't run promptly.
                # os._exit terminates immediately with a non-zero status.
                msg = (
                    f"\n[mem] FATAL: system memory usage {percent:.1f}% "
                    f"({used_gb:.2f}/{total_gb:.2f} GB) exceeded the "
                    f"{kill_threshold_percent:.1f}% threshold; killing the process.\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                sys.stdout.flush()
                os._exit(1)
        time.sleep(interval_seconds)

def start_memory_logger(interval_seconds: int = 10, kill_threshold_percent: float = 95.0):
    """Start a background thread that logs system memory usage every
    ``interval_seconds`` and hard-kills the process (os._exit(1)) if usage reaches
    ``kill_threshold_percent``.  Pass ``kill_threshold_percent=None`` to log only.
    """
    t = threading.Thread(
        target=_memory_logger_loop,
        args=(interval_seconds, kill_threshold_percent),
        daemon=True,
    )
    t.start()

