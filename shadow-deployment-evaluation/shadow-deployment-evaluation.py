import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    total = len(production_log)
    if total == 0:
        return {"promote": False, "metrics": {}}

    prod_correct = 0
    shadow_correct = 0
    agreements = 0
    latencies = []

    for p, s in zip(production_log, shadow_log):
        if p.get("prediction") == p.get("actual"):
            prod_correct += 1
        if s.get("prediction") == s.get("actual"):
            shadow_correct += 1
        if p.get("prediction") == s.get("prediction"):
            agreements += 1
        if "latency_ms" in s:
            latencies.append(s["latency_ms"])

    production_accuracy = prod_correct / total
    shadow_accuracy = shadow_correct / total
    accuracy_gain = shadow_accuracy - production_accuracy
    agreement_rate = agreements / total

    latencies.sort()
    idx = int(math.ceil(0.95 * len(latencies))) - 1 if latencies else 0
    shadow_latency_p95 = latencies[max(0, idx)] if latencies else 0

    promote = True
    if "max_latency_p95" in criteria and shadow_latency_p95 > criteria["max_latency_p95"]:
        promote = False
    if "min_accuracy_gain" in criteria and accuracy_gain < criteria["min_accuracy_gain"]:
        promote = False
    if "min_agreement_rate" in criteria and agreement_rate < criteria["min_agreement_rate"]:
        promote = False

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate
        }
    }