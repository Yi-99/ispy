# test_prediction.py - Enhanced version with JSON output storage
import base64
import requests
import json
import os
from pathlib import Path
from datetime import datetime

def ensure_output_directory():
    """Create the predictions/outputs directory if it doesn't exist"""
    output_dir = Path("predictions/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_prediction_result(image_path, result, expected_label=None):
    """Save individual prediction result to JSON file"""
    output_dir = ensure_output_directory()
    
    # Create filename from image path
    image_name = Path(image_path).stem
    output_file = output_dir / f"{image_name}_prediction.json"
    
    # Prepare the complete result data
    prediction_data = {
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "expected_label": expected_label,
        "prediction_result": result,
        "model_info": {
            "model_used": result.get("model_used", "resnet50"),
            "processing_time": result.get("processing_time", 0.0)
        },
        "fraud_analysis": {
            "fraud_probability": result.get("fraud_probability", 0.0),
            "non_fraud_probability": 1 - result.get("fraud_probability", 0.0),
            "risk_level": result.get("risk_level", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "uncertainty": result.get("uncertainty", None)
        },
        "prediction_summary": {
            "is_fraud_predicted": result.get("fraud_probability", 0.0) > 0.5,
            "prediction_correct": None  # Will be calculated if expected_label provided
        }
    }
    
    # Calculate prediction correctness if expected label is provided
    if expected_label:
        predicted_fraud = result.get("fraud_probability", 0.0) > 0.5
        expected_fraud = expected_label.lower() == 'fraud'
        prediction_data["prediction_summary"]["prediction_correct"] = predicted_fraud == expected_fraud
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"💾 Saved prediction to: {output_file}")
    return output_file

def test_single_image(image_path, expected_label=None):
    """Test a single image and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    if expected_label:
        print(f"Expected: {expected_label}")
    print(f"{'='*60}")
    
    try:
        # Convert image to base64
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        # Test API
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "image_data": encoded,
                "model_name": "resnet50"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing Time: {result['processing_time']:.4f}s")
            
            # Save individual result to JSON
            output_file = save_prediction_result(image_path, result, expected_label)
            
            # Check if prediction matches expected
            if expected_label:
                predicted_fraud = result['fraud_probability'] > 0.5
                expected_fraud = expected_label.lower() == 'fraud'
                match = predicted_fraud == expected_fraud
                print(f"🎯 Prediction Match: {'✅ CORRECT' if match else '❌ INCORRECT'}")
                print(f"   Expected: {'Fraud' if expected_fraud else 'Non-Fraud'}")
                print(f"   Predicted: {'Fraud' if predicted_fraud else 'Non-Fraud'}")
            
            return result
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def create_summary_report(results, output_dir):
    """Create a summary report of all predictions"""
    summary_file = output_dir / "prediction_summary.json"
    
    # Calculate statistics
    total_images = len(results)
    correct_predictions = sum(1 for r in results if r.get('prediction_correct', False))
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    fraud_probs = [r['fraud_prob'] for r in results]
    
    # Risk level distribution
    risk_levels = {}
    for r in results:
        risk = r['risk_level']
        risk_levels[risk] = risk_levels.get(risk, 0) + 1
    
    summary_data = {
        "summary_info": {
            "total_images_tested": total_images,
            "correct_predictions": correct_predictions,
            "accuracy_percentage": round(accuracy * 100, 1),
            "timestamp": datetime.now().isoformat()
        },
        "fraud_probability_stats": {
            "min_fraud_probability": round(min(fraud_probs), 4),
            "max_fraud_probability": round(max(fraud_probs), 4),
            "average_fraud_probability": round(sum(fraud_probs) / len(fraud_probs), 4)
        },
        "risk_level_distribution": risk_levels,
        "individual_results": results
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"📊 Summary report saved to: {summary_file}")
    return summary_file

def main():
    """Test multiple images from different categories"""
    print("🚗 Car Insurance Fraud Detection - Multi-Image Test with JSON Output")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = ensure_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Test images from different categories
    test_cases = [
        # Fraud cases (should have high fraud probability)
        ("data/test/Fraud/63.jpg", "Fraud"),
        ("data/test/Fraud/385.jpg", "Fraud"),
        ("data/test/Fraud/1028.jpg", "Fraud"),
        ("data/test/Fraud/1060.jpg", "Fraud"),
        ("data/test/Fraud/1227.jpg", "Fraud"),
        
        # Non-Fraud cases (should have low fraud probability)
        ("data/test/Non-Fraud/3.jpg", "Non-Fraud"),
        ("data/test/Non-Fraud/4.jpg", "Non-Fraud"),
    ]
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for image_path, expected_label in test_cases:
        if os.path.exists(image_path):
            result = test_single_image(image_path, expected_label)
            if result:
                # Check accuracy
                predicted_fraud = result['fraud_probability'] > 0.5
                expected_fraud = expected_label.lower() == 'fraud'
                prediction_correct = predicted_fraud == expected_fraud
                
                if prediction_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                results.append({
                    'image': image_path,
                    'expected': expected_label,
                    'fraud_prob': result['fraud_probability'],
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence'],
                    'prediction_correct': prediction_correct
                })
        else:
            print(f"⚠️  File not found: {image_path}")
    
    # Create summary report
    if results:
        create_summary_report(results, output_dir)
    
    # Console summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images Tested: {len(results)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions/total_predictions*100:.1f}%" if total_predictions > 0 else "N/A")
    
    if results:
        fraud_probs = [r['fraud_prob'] for r in results]
        print(f"\nFraud Probability Range: {min(fraud_probs):.4f} - {max(fraud_probs):.4f}")
        print(f"Average Fraud Probability: {sum(fraud_probs)/len(fraud_probs):.4f}")
        
        # Show risk level distribution
        risk_levels = {}
        for r in results:
            risk = r['risk_level']
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
        
        print(f"\nRisk Level Distribution:")
        for risk, count in risk_levels.items():
            print(f"  {risk}: {count} images")

if __name__ == "__main__":
    main()