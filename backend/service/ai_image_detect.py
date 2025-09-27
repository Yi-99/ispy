import os
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider='auto',
    api_key='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'
)

def detect_image(image_path):
    """
    Detect objects in an image using Hugging Face model
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Formatted detection results
    """
    try:
        # Image Classification to detect the image
        output = client.image_classification(image_path, model="haywoodsloan/ai-image-detector-deploy")
        
        # Format the output to a string
        formatted_parts = [f"{item['label']} score = {item['score']}" for item in output]
        final_string = ", ".join(formatted_parts)

        print(final_string)
        return final_string
    except Exception as e:
        print(f"Error in detect_image: {e}")
        return f"Error: {str(e)}"

# Test function
if __name__ == "__main__":
    # Test with a sample image path
    test_image_path = "../test_image.png"
    result = detect_image(test_image_path)
    print(f"Detection result: {result}")