import os
import sys
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cnnClassifier.pipeline.predict import PredictionPipeline


def run_test():
    print("=== Testing Chicken Disease Prediction Pipeline ===")
    test_img_path = "test_fecal_sample.jpg"
    
    # Create a synthetic test image if not present
    if not os.path.exists(test_img_path):
        img = Image.new('RGB', (224, 224), color=(130, 100, 50))
        img.save(test_img_path)
        print(f"Created sample test image: {test_img_path}")

    pipeline = PredictionPipeline(test_img_path)
    results = pipeline.predict()
    
    print("\nPrediction Results:")
    for res in results:
        print(f"  Condition Predicted : {res.get('prediction')}")
        print(f"  Confidence Score    : {res.get('confidence')}")
        print(f"  Execution Status    : {res.get('status')}")

    print("\n[SUCCESS] Prediction Pipeline verification completed!")


if __name__ == "__main__":
    run_test()
