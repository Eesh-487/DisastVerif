import json
from decision_maker import final_decision

def generate_report(confidences, evidence_list):
    """
    Generate a JSON formatted report based on the decision logic.
    """
    result = final_decision(confidences, evidence_list)
    return json.dumps(result, indent=4)

if __name__ == "__main__":
    # Example usage for testing the report generation
    confidences = [0.8, 0.65, 0.9]
    evidence_list = [
        {"detail": "Sensor A reported anomaly", "score": 0.8},
        {"detail": "Sensor B reported normal operation", "score": 0.65},
        {"detail": "Sensor C reported anomaly", "score": 0.9}
    ]
    report = generate_report(confidences, evidence_list)
    print(report)
    