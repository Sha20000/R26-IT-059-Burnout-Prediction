def _categorize(score: float, thresholds: dict) -> str:
    if score <= thresholds["low_max"]:
        return "Low"
    if score <= thresholds["medium_max"]:
        return "Medium"
    return "High"


def compute_final(record: dict, config: dict) -> dict:
    weights = config["weights"]
    thresholds = config["thresholds"]
    override = config["override"]

    scores = {}
    for key, weight in weights.items():
        value = record.get(key)
        if isinstance(value, (int, float)):
            scores[key] = float(value)

    if not scores:
        return {
            "final_score": None,
            "final_risk": "Unknown",
            "override_applied": False,
            "contributions": {},
            "normalized_weights": {},
        }

    total_weight = sum(weights[key] for key in scores)
    weighted = sum(scores[key] * weights[key] for key in scores) / total_weight

    final_risk = _categorize(weighted, thresholds)

    override_applied = False
    academic_score = record.get("academic_score")
    engagement_score = record.get("engagement_score")
    if isinstance(academic_score, (int, float)) and isinstance(engagement_score, (int, float)):
        if academic_score > override["academic_min"] and engagement_score > override["engagement_min"]:
            override_applied = True
            final_risk = "High"

    contributions = {key: round(scores[key] * weights[key], 4) for key in scores}
    normalized_weights = {key: round(weights[key] / total_weight, 4) for key in scores}

    return {
        "final_score": round(weighted, 4),
        "final_risk": final_risk,
        "override_applied": override_applied,
        "contributions": contributions,
        "normalized_weights": normalized_weights,
    }
