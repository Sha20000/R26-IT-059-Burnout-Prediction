def _advisor_recommendation(risk: str) -> str:
    if risk == "High":
        return "Immediate advisor outreach recommended. Schedule a 1:1 check-in within 7 days."
    if risk == "Medium":
        return "Monitor closely. Encourage support resources and follow-up within 2 weeks."
    if risk == "Low":
        return "Maintain current support. Provide periodic encouragement and check-ins."
    return "Insufficient data to provide a recommendation."


def build_explanation(record: dict, meta: dict) -> dict:
    final_score = meta.get("final_score")
    final_risk = meta.get("final_risk")
    override_applied = meta.get("override_applied")

    contributions = meta.get("contributions", {})
    ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    top_models = [name.replace("_score", "") for name, _ in ranked[:2]]

    reasons = []
    for key, value in record.items():
        if key.endswith("_reason") and value:
            reasons.append(value)

    if final_score is None:
        summary = "No final score available due to missing model outputs."
    else:
        summary = f"Final risk score is {final_score:.2f} ({final_risk})."
        if top_models:
            summary += f" Primary drivers: {', '.join(top_models)}."

    if override_applied:
        summary += " Override rule applied due to high academic and engagement risk signals."

    return {
        "summary": summary,
        "model_reasons": reasons,
        "advisor_recommendation": _advisor_recommendation(final_risk),
    }
