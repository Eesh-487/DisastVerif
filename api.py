from flask import Flask, request, jsonify
from verification_service import verify_disaster,load_models
import os
import traceback
import json
from flask_cors import CORS
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)  # Enable CORS for all routes
# Add this helper function before your route definitions
def sanitize_for_json(obj):
    """Convert NumPy objects to standard Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
# API endpoint for disaster verification and submission
@app.route('/api/disasters', methods=['POST'])
def create_disaster():
    try:
        # Get disaster data from request, handling different content types
        if request.is_json:
            disaster_data = request.json
        else:
            # Try to parse the data even if Content-Type is not set correctly
            try:
                disaster_data = json.loads(request.data)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': 'Invalid JSON data',
                    'error': str(e)
                }), 400
        
        # Ensure all required fields are present
        required_fields = ['latitude', 'longitude', 'description', 'location', 'disaster_type', 'date']
        for field in required_fields:
            if field not in disaster_data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400
        
        # In your create_disaster route, modify the verification result processing:
# Verify the disaster
        verification_result = verify_disaster(disaster_data)
        # Sanitize the result for JSON
        verification_result = sanitize_for_json(verification_result)
# Sanitize the result for JSO
        # Only save to database if probability is above 50%
        if verification_result['probability'] >= 0.5:
            # Add verification result to disaster data (optional)
            disaster_data['verification'] = {
                'probability': verification_result['probability'],
                # 'classification': verification_result['classification'],
                # 'evidence': verification_result['evidence']
            }
            
            # TODO: Save to database
            # This is where you would implement your database saving logic
            # For now, we'll just simulate a successful save
            
            return jsonify({
                'success': True,
                'message': 'Disaster verified and saved',
                'verification': verification_result
            }), 201
        else:
            # Disaster failed verification
            return jsonify({
                'success': False,
                'message': 'Disaster verification failed',
                'verification': verification_result
            }), 403
    
    except Exception as e:
        # Log the error
        print(f"Error processing disaster report: {str(e)}")
        print(traceback.format_exc())
        
        # Return error response
        return jsonify({
            'success': False,
            'message': 'An error occurred while processing the disaster report',
            'error': str(e)
        }), 500

# API endpoint to get verification status only (without saving)
@app.route('/api/verify-disaster', methods=['POST'])
def verify_disaster_only():
    try:
        # Similar to above, handle different content types
        if request.is_json:
            disaster_data = request.json
        else:
            try:
                disaster_data = json.loads(request.data)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': 'Invalid JSON data',
                    'error': str(e)
                }), 400
                
        verification_result = verify_disaster(disaster_data)# Sanitize the result for JSON
        verification_result = sanitize_for_json(verification_result)
        
        return jsonify({
            'success': True,
            'verification': verification_result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Error during verification',
            'error': str(e)
        }), 500

# Simple health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Disaster Verification API'
    }), 200
def initialize_app():
    """Load models at startup to avoid retraining for each request"""
    print("Pre-loading verification models...")
    load_models()
    print("Verification service ready")
if __name__ == '__main__':
    # Initialize the app (load models)
    initialize_app()
    
    # Run the API server
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Disaster Verification API on port {port}...")
    print(f"Health check: http://localhost:{port}/api/health")
    print(f"Verification endpoint: http://localhost:{port}/api/verify-disaster")
    print(f"Disaster submission endpoint: http://localhost:{port}/api/disasters")
    app.run(host='0.0.0.0', port=port, debug=True)