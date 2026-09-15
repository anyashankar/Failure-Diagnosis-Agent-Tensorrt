import os
from datetime import datetime

NOTIFICATIONS_LOG = os.path.expanduser("~/notifications.log")

def send_notification(model_name: str, success: bool, summary: dict):
    """
    Sends a completion notification. Currently writes to a local log file
    with a clear terminal banner — swap the body of this function for a
    real webhook (Slack, email, etc.) without touching any caller code.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    status_line = "SUCCESS" if success else "FAILED"

    lines = [
        f"[{timestamp}] Build Agent Notification — {model_name}: {status_line}",
    ]

    if success and summary:
        bench = summary.get("benchmark", {})
        acc = summary.get("accuracy", {})
        lines.append(
            f"  Latency: avg={bench.get('avg_latency_ms', 'N/A'):.3f}ms"
            if isinstance(bench.get('avg_latency_ms'), (int, float))
            else f"  Latency: {bench}"
        )
        lines.append(
            f"  Accuracy: worst_cosine_similarity={acc.get('worst_cosine_similarity', 'N/A'):.6f}, passed={acc.get('passed')}"
            if isinstance(acc.get('worst_cosine_similarity'), (int, float))
            else f"  Accuracy: {acc}"
        )
    elif not success:
        if summary and summary.get("escalation_reason"):
            lines.append("  Agent could not find an automated fix. Best-effort diagnosis for review:")
            lines.append(f"    {summary['escalation_reason']}")
        else:
            lines.append("  Agent could not resolve the build failure — human review needed.")

    message = "\n".join(lines)

    print("\n" + "=" * 60)
    print(message)
    print("=" * 60 + "\n")

    with open(NOTIFICATIONS_LOG, "a") as f:
        f.write(message + "\n\n")

    return message
