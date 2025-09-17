import json
import asyncio
from pathlib import Path
from services.ocr_processor import OCRProcessor


async def process_images_in_folder(folder_path: str = "images"):
    """Process all images in the specified folder using OCR pipeline"""
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png'}
    
    # Get all image files in folder
    image_folder = Path(folder_path)
    if not image_folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    image_files = [f for f in image_folder.iterdir() 
                  if f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"No supported image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} image(s) to process:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    print()
    
    # Process each image
    results = {}
    
    for img_file in image_files:
        print(f"Processing {img_file.name}...")
        
        try:
            # Read image file
            with open(img_file, 'rb') as f:
                image_bytes = f.read()
            
            # Process through OCR pipeline
            result = await processor.process_image(image_bytes)
            
            # Store result
            results[img_file.name] = {
                'rekognition_texts': [{'text': t.text, 'confidence': t.confidence} 
                                    for t in result.rekognition],
                'bedrock_texts': [{'text': t.text, 'confidence': t.confidence} 
                                for t in result.bedrock],
                'second_opinion_triggered': result.second_opinion.triggered,
                'second_opinion_reason': result.second_opinion.reason if result.second_opinion.triggered else None
            }
            
            # Print summary
            print(f"  Rekognition found: {len(result.rekognition)} text items")
            if result.bedrock:
                print(f"  Bedrock found: {len(result.bedrock)} text items")
            if result.second_opinion.triggered:
                print(f"  Second opinion triggered: {result.second_opinion.reason}")
            print()
            
        except Exception as e:
            print(f"  Error processing {img_file.name}: {str(e)}")
            results[img_file.name] = {'error': str(e)}
            print()
    
    # Save results to JSON file
    output_file = "ocr_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    successful = len([r for r in results.values() if 'error' not in r])
    failed = len(results) - successful
    print(f"\nSummary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    asyncio.run(process_images_in_folder())