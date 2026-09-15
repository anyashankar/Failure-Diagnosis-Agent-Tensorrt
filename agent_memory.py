import json
import re
import hashlib
import os

MEMORY_PATH = os.path.expanduser("~/agent_memory.json")

def _normalize_log(log_text: str) -> str:
    # Strip volatile parts (timestamps, addresses, numbers) so the same
    # underlying error signature matches across different runs/models.
    text = re.sub(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\]", "", log_text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\d+", "N", text)
    return text.strip()

def _signature(log_text: str) -> str:
    normalized = _normalize_log(log_text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

def _load_memory() -> dict:
    if not os.path.exists(MEMORY_PATH):
        return {}
    try:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_memory(memory: dict):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

def get_cached_fix(log_text: str):
    """Returns the tool_name previously confirmed to fix this error signature, or None on a miss."""
    memory = _load_memory()
    entry = memory.get(_signature(log_text))
    return entry.get("tool_name") if entry else None

def record_fix(log_text: str, tool_name: str):
    """Records that tool_name successfully resolved this error signature — persists across runs."""
    memory = _load_memory()
    sig = _signature(log_text)
    entry = memory.get(sig, {"tool_name": tool_name, "seen_count": 0})
    entry["tool_name"] = tool_name
    entry["seen_count"] = entry.get("seen_count", 0) + 1
    memory[sig] = entry
    _save_memory(memory)
