THRESHOLD = 0.7

def compute_final_confidence(confidences):
    """
    Compute the mean confidence from a list of values.
    """
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)

def generate_supporting_evidence(evidence_list):
    """
    Create a list of formatted evidence details.
    """
    evidence_details = []
    for evidence in evidence_list:
        detail = f"{evidence.get('detail', 'No detail provided')} (score: {evidence.get('score', 0)})"
        evidence_details.append(detail)
    return evidence_details

def final_decision(confidences, evidence_list):
    """
    Determine the final verification decision based on confidence scores.

    Parameters:
        confidences (list of float): Confidence scores from various sources.
        evidence_list (list of dict): Evidence dictionaries with 'detail' and 'score'.
    
    Returns:
        dict: A dictionary containing the final confidence, decision, and supporting evidence.
    """
    final_conf = compute_final_confidence(confidences)
    evidence = generate_supporting_evidence(evidence_list)
    decision = "Verified" if final_conf >= THRESHOLD else "Not Verified"
    return {
        "final_confidence": final_conf,
        "decision": decision,
        "supporting_evidence": evidence
    }