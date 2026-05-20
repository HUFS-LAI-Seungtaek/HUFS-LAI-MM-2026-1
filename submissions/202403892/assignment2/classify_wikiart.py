#!/usr/bin/env python3
"""
WikiArt Image Classification using Schema-Guided Decoding
Uses OpenRouter API with Claude to classify images according to specified schema
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from datasets import load_dataset
import hashlib

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OUTPUT_DIR = Path(__file__).parent
IMAGES_DIR = OUTPUT_DIR / "images"

# JSON Schema for guided decoding
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image contains human figures or human-like entities"
        },
        "has_animal": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image contains animals or animal-like creatures"
        },
        "has_flower": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image contains flowers or floral elements"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief natural language explanation of classifications, including location descriptors (center, top-left, foreground, background, etc.)"
        }
    },
    "required": ["has_human", "has_animal", "has_flower", "reasoning"],
    "additionalProperties": False
}

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path: str, image_filename: str) -> Optional[dict]:
    """
    Classify a single image using OpenRouter API with schema-guided decoding
    
    Args:
        image_path: Path to the image file
        image_filename: Name of the image file
        
    Returns:
        Dictionary with classification results or None if failed
    """
    try:
        # Encode image to base64
        image_data = encode_image_to_base64(image_path)
        
        # Prepare the system prompt with clear definitions
        system_prompt = """You are an expert art image classifier. Analyze the provided artwork image and classify it according to these definitions:

- **has_human**: Set to "yes" if the image contains any human figures, faces, silhouettes, or clearly human-like entities (including statues, paintings of humans, etc.). Set to "no" if no human representations are visible.

- **has_animal**: Set to "yes" if the image contains any animals, insects, birds, mythical creatures, or animal-like figures (including depictions in paintings, sculptures, etc.). Set to "no" if no animal representations are visible.

- **has_flower**: Set to "yes" if the image contains flowers, floral arrangements, or botanical flower elements (including paintings/depictions of flowers). Set to "no" if no flower elements are visible.

For the reasoning field, provide a brief explanation that includes:
1. What you observed in the image
2. Location descriptors where applicable (center, top-left, foreground, background, etc.)
3. Why you classified each category as yes/no

Be precise and specific in your reasoning."""

        user_prompt = f"""Please analyze this artwork image and classify it according to the provided schema. 
        
Image: {image_filename}

Respond with only valid JSON matching the specified schema."""

        # Prepare the request with schema-guided decoding
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image",
                            "image": image_data
                        }
                    ]
                }
            ],
            # Schema-guided decoding configuration
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ImageClassification",
                    "schema": CLASSIFICATION_SCHEMA,
                    "strict": True
                }
            }
        }
        
        # Make the API request
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"  ❌ API Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        
        # Parse the response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            classification = json.loads(content)
            return classification
        else:
            print(f"  ❌ Unexpected response format: {result}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing error: {e}")
        return None
    except requests.RequestException as e:
        print(f"  ❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None

def download_wikiart_images(num_images: int = 20) -> list:
    """
    Download WikiArt images from Hugging Face dataset
    
    Args:
        num_images: Number of images to download
        
    Returns:
        List of paths to downloaded images
    """
    print(f"📥 Loading WikiArt dataset...")
    
    try:
        # Load the WikiArt dataset
        dataset = load_dataset("huggan/wikiart", split="train", streaming=False)
        
        # Create images directory if it doesn't exist
        IMAGES_DIR.mkdir(exist_ok=True, parents=True)
        
        downloaded_images = []
        
        # Download first num_images images
        for idx in range(min(num_images, len(dataset))):
            sample = dataset[idx]
            image = sample["image"]
            
            # Generate filename with hash of index to avoid conflicts
            filename = f"wikiart_{idx:04d}.jpg"
            filepath = IMAGES_DIR / filename
            
            # Save image
            image.save(str(filepath))
            downloaded_images.append(filepath)
            
            print(f"  ✅ Downloaded: {filename}")
        
        return downloaded_images
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

def classify_images(image_paths: list) -> list:
    """
    Classify a list of images
    
    Args:
        image_paths: List of paths to images
        
    Returns:
        List of classification results with metadata
    """
    results = []
    total = len(image_paths)
    
    print(f"\n🔍 Classifying {total} images...\n")
    
    for idx, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).name
        print(f"[{idx}/{total}] Processing: {filename}")
        
        classification = classify_image(str(image_path), filename)
        
        if classification:
            result = {
                "image": filename,
                "image_path": str(image_path),
                **classification
            }
            results.append(result)
            print(f"  ✅ Classification complete")
            print(f"     Human: {classification['has_human']}, Animal: {classification['has_animal']}, Flower: {classification['has_flower']}")
        else:
            print(f"  ⏭️  Skipped due to error")
        
        print()
    
    return results

def generate_html_report(results: list, output_file: str = "results.html"):
    """
    Generate an HTML report of classification results
    
    Args:
        results: List of classification results
        output_file: Path to output HTML file
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WikiArt Image Classification Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .stat-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stat-box h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-box .number {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .results {
            padding: 30px 20px;
        }
        
        .result-card {
            background: white;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .result-title {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }
        
        .result-index {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        
        .classifications {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .classification {
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
            text-align: center;
        }
        
        .classification-label {
            font-size: 0.85em;
            color: #666;
            font-weight: 600;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        
        .classification-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .yes {
            color: #27ae60;
        }
        
        .no {
            color: #e74c3c;
        }
        
        .reasoning {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            line-height: 1.6;
            color: #444;
            font-style: italic;
            border-left: 3px solid #667eea;
        }
        
        .image-thumbnail {
            width: 100%;
            max-width: 300px;
            margin: 15px 0;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 WikiArt Image Classification</h1>
            <p>Schema-Guided Decoding with OpenRouter API</p>
        </div>
"""
    
    # Add statistics
    total_images = len(results)
    human_count = sum(1 for r in results if r.get("has_human") == "yes")
    animal_count = sum(1 for r in results if r.get("has_animal") == "yes")
    flower_count = sum(1 for r in results if r.get("has_flower") == "yes")
    
    html_content += f"""
        <div class="stats">
            <div class="stat-box">
                <h3>Total Images</h3>
                <div class="number">{total_images}</div>
            </div>
            <div class="stat-box">
                <h3>With Human</h3>
                <div class="number">{human_count}</div>
            </div>
            <div class="stat-box">
                <h3>With Animal</h3>
                <div class="number">{animal_count}</div>
            </div>
            <div class="stat-box">
                <h3>With Flower</h3>
                <div class="number">{flower_count}</div>
            </div>
        </div>
        
        <div class="results">
"""
    
    # Add result cards
    for idx, result in enumerate(results, 1):
        image_filename = result.get("image", "unknown")
        
        html_content += f"""
            <div class="result-card">
                <div class="result-header">
                    <div class="result-title">
                        Image #{idx}
                    </div>
                    <div class="result-index">{image_filename}</div>
                </div>
                
                <div class="classifications">
                    <div class="classification">
                        <div class="classification-label">Human</div>
                        <div class="classification-value {result.get('has_human', 'no').lower()}">{result.get("has_human", "unknown").upper()}</div>
                    </div>
                    <div class="classification">
                        <div class="classification-label">Animal</div>
                        <div class="classification-value {result.get('has_animal', 'no').lower()}">{result.get("has_animal", "unknown").upper()}</div>
                    </div>
                    <div class="classification">
                        <div class="classification-label">Flower</div>
                        <div class="classification-value {result.get('has_flower', 'no').lower()}">{result.get("has_flower", "unknown").upper()}</div>
                    </div>
                </div>
                
                <div class="reasoning">
                    <strong>Reasoning:</strong> {result.get("reasoning", "No reasoning provided")}
                </div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="footer">
            <p>Generated by WikiArt Classification System | Using Schema-Guided Decoding</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    output_path = OUTPUT_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")

def save_results_json(results: list, output_file: str = "results.json"):
    """Save results as JSON for later reference"""
    output_path = OUTPUT_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON results saved: {output_path}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("🎨 WikiArt Image Classification using Schema-Guided Decoding")
    print("=" * 60)
    print()
    
    # Check API key
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not found in environment variables")
        print("   Please create a .env file with: OPENROUTER_API_KEY=your_key")
        return
    
    # Download images
    print("Step 1: Downloading WikiArt images")
    print("-" * 60)
    image_paths = download_wikiart_images(num_images=20)
    
    if not image_paths:
        print("❌ No images were downloaded")
        return
    
    print(f"✅ Successfully downloaded {len(image_paths)} images")
    
    # Classify images
    print("\nStep 2: Classifying images")
    print("-" * 60)
    results = classify_images(image_paths)
    
    if not results:
        print("❌ No classification results obtained")
        return
    
    print(f"✅ Successfully classified {len(results)} images")
    
    # Generate reports
    print("\nStep 3: Generating reports")
    print("-" * 60)
    generate_html_report(results)
    save_results_json(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Classification completed successfully!")
    print("=" * 60)
    print(f"Total images classified: {len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print()

if __name__ == "__main__":
    main()
