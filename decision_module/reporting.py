"""
Module for generating verification reports and making final decisions.
This module formats the output from the verification process into readable reports.
"""

import json
from datetime import datetime
import os

# Constants
VERIFICATION_THRESHOLDS = {
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4,
    "VERY_LOW": 0.0
}

def generate_report(confidences, evidence, report_format="text"):
    """
    Generate a verification report based on model confidences and evidence.
    
    Args:
        confidences (list): List of confidence scores from different models
        evidence (list): List of evidence dictionaries with details and scores
        report_format (str): Format of the report (text, json, html)
        
    Returns:
        str: Formatted report
    """
    # Calculate overall verification score (average of confidences)
    if confidences:
        overall_score = sum(confidences) / len(confidences)
    else:
        overall_score = 0.0
    
    # Determine verification status
    if overall_score >= VERIFICATION_THRESHOLDS["HIGH"]:
        status = "VERIFIED"
        explanation = "This disaster event is verified with high confidence."
    elif overall_score >= VERIFICATION_THRESHOLDS["MEDIUM"]:
        status = "LIKELY"
        explanation = "This disaster event is likely to be valid, but with some uncertainty."
    elif overall_score >= VERIFICATION_THRESHOLDS["LOW"]:
        status = "UNCERTAIN"
        explanation = "There is insufficient evidence to verify this disaster event."
    else:
        status = "UNVERIFIED"
        explanation = "This disaster event could not be verified with available evidence."
    
    # Build report data
    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verification_status": status,
        "overall_confidence": overall_score,
        "explanation": explanation,
        "evidence": evidence,
        "model_confidences": {
            "geospatial": confidences[0] if len(confidences) > 0 else None,
            "text": confidences[1] if len(confidences) > 1 else None,
            "news": confidences[2] if len(confidences) > 2 else None
        }
    }
    
    # Format report based on requested format
    if report_format.lower() == "json":
        return json.dumps(report_data, indent=2)
    elif report_format.lower() == "html":
        return _generate_html_report(report_data)
    else:  # Default to text format
        return _generate_text_report(report_data)

def _generate_text_report(report_data):
    """Generate a text-based report"""
    
    lines = [
        "====================================================",
        "               DISASTER VERIFICATION REPORT         ",
        "====================================================",
        f"Timestamp: {report_data['timestamp']}",
        "",
        f"VERIFICATION STATUS: {report_data['verification_status']}",
        f"Overall Confidence: {report_data['overall_confidence']:.4f}",
        "",
        "EXPLANATION:",
        report_data['explanation'],
        "",
        "MODEL CONFIDENCES:",
        f"- Geospatial Analysis: {report_data['model_confidences']['geospatial']:.4f}" if report_data['model_confidences']['geospatial'] is not None else "- Geospatial Analysis: N/A",
        f"- Text Similarity: {report_data['model_confidences']['text']:.4f}" if report_data['model_confidences']['text'] is not None else "- Text Similarity: N/A",
        f"- News Verification: {report_data['model_confidences']['news']:.4f}" if report_data['model_confidences']['news'] is not None else "- News Verification: N/A",
        "",
        "SUPPORTING EVIDENCE:"
    ]
    
    # Add evidence items
    for i, item in enumerate(report_data['evidence'], 1):
        lines.append(f"{i}. {item['detail']}: {item['score']:.4f}")
    
    lines.append("")
    lines.append("====================================================")
    
    return "\n".join(lines)

def _generate_html_report(report_data):
    """Generate an HTML-based report"""
    
    # Define CSS style
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .report { max-width: 800px; margin: 0 auto; }
        .header { background-color: #2c3e50; color: white; padding: 10px; text-align: center; }
        .status { font-size: 24px; font-weight: bold; margin: 20px 0; text-align: center; }
        .status-VERIFIED { color: #27ae60; }
        .status-LIKELY { color: #f39c12; }
        .status-UNCERTAIN { color: #e67e22; }
        .status-UNVERIFIED { color: #e74c3c; }
        .section { margin: 20px 0; }
        .section-title { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        .evidence-item { margin: 10px 0; padding: 5px; }
        .confidence { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
        .footer { margin-top: 30px; font-size: 12px; color: #7f8c8d; text-align: center; }
    </style>
    """
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Disaster Verification Report</title>
        {css}
    </head>
    <body>
        <div class="report">
            <div class="header">
                <h1>Disaster Verification Report</h1>
                <p>{report_data['timestamp']}</p>
            </div>
            
            <div class="status status-{report_data['verification_status']}">
                {report_data['verification_status']}
                <div>Overall Confidence: {report_data['overall_confidence']:.4f}</div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Explanation</h2>
                <p>{report_data['explanation']}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Model Confidences</h2>
                <div class="confidence">
    """
    
    # Add model confidences
    for model, score in report_data['model_confidences'].items():
        if score is not None:
            html += f"<div><strong>{model.title()} Analysis:</strong> {score:.4f}</div>"
        else:
            html += f"<div><strong>{model.title()} Analysis:</strong> N/A</div>"
    
    html += """
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Supporting Evidence</h2>
    """
    
    # Add evidence items
    for i, item in enumerate(report_data['evidence'], 1):
        html += f"""
                <div class="evidence-item">
                    <strong>{i}.</strong> {item['detail']}: <span>{item['score']:.4f}</span>
                </div>
        """
    
    html += """
            </div>
            
            <div class="footer">
                Generated by Disaster Verifier System
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def save_report_to_file(report, filename=None, format="text"):
    """
    Save the verification report to a file.
    
    Args:
        report (str): The formatted report
        filename (str): Name of the file to save (if None, a default name is generated)
        format (str): Format of the report (text, json, html)
        
    Returns:
        str: Path to the saved file
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Generate default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"disaster_report_{timestamp}.{format}"
    
    file_path = os.path.join(reports_dir, filename)
    
    # Write report to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return file_path

def classify_disaster_probability(geospatial_score, text_score, news_score):
    """
    Classify the probability of a disaster based on model scores.
    
    Args:
        geospatial_score (float): Score from geospatial analysis (0-1)
        text_score (float): Score from text similarity analysis (0-1)
        news_score (float): Score from news verification (0-1)
        
    Returns:
        tuple: (probability_class, weighted_score)
    """
    # Define weights for each model
    weights = {
        'geospatial': 0.4,
        'text': 0.3,
        'news': 0.3
    }
    
    # Calculate weighted average
    weighted_score = (
        geospatial_score * weights['geospatial'] +
        text_score * weights['text'] +
        news_score * weights['news']
    )
    
    # Determine probability class
    if weighted_score >= 0.8:
        probability_class = "HIGH PROBABILITY"
    elif weighted_score >= 0.5:
        probability_class = "MEDIUM PROBABILITY"
    elif weighted_score >= 0.3:
        probability_class = "LOW PROBABILITY"
    else:
        probability_class = "VERY LOW PROBABILITY"
    
    return probability_class, weighted_score