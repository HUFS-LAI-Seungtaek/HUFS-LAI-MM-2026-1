import os
import json
import base64
import requests
from io import BytesIO
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# .env 파일 로드
load_dotenv()

# 설정
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.0-flash-001" # 비전 성능이 뛰어난 모델
IMAGE_SAVE_DIR = "images"
SAMPLE_COUNT = 20

# 폴더 생성
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def encode_image(image):
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def classify_image(image, index):
    base64_image = encode_image(image)
    
    prompt = """
    Classify this WikiArt image based on the presence of humans, animals, or flowers.
    - has_human: 'yes' if there are people, statues of people, or distinct human body parts.
    - has_animal: 'yes' if any animals are present.
    - has_flower: 'yes' if flowers are visible.
    
    Provide a short reasoning in English, using spatial terms like 'center', 'top-left', 'foreground', or 'background'.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "has_human": {"type": "string", "enum": ["yes", "no"]},
                        "has_animal": {"type": "string", "enum": ["yes", "no"]},
                        "has_flower": {"type": "string", "enum": ["yes", "no"]},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["has_human", "has_animal", "has_flower", "reasoning"]
                }
            }
        }
    )
    return json.loads(response.choices[0].message.content)

def main():
    print("🚀 WikiArt 데이터셋 로딩 중...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    samples = list(ds.take(SAMPLE_COUNT))
    
    results = []
    
    print(f"🖼️ {SAMPLE_COUNT}개의 이미지 분류 시작...")
    for i, item in enumerate(samples):
        img = item['image']
        img_filename = f"wikiart_{i}.jpg"
        img_path = os.path.join(IMAGE_SAVE_DIR, img_filename)
        
        # 이미지 저장
        img.convert("RGB").save(img_path)
        
        # AI 분류
        try:
            prediction = classify_image(img, i)
            prediction['image_path'] = f"images/{img_filename}"
            results.append(prediction)
            print(f"[{i+1}/{SAMPLE_COUNT}] 완료: {img_filename}")
        except Exception as e:
            print(f"[{i+1}/{SAMPLE_COUNT}] 오류 발생: {e}")

    # HTML 생성
    generate_html(results)
    print("✅ 모든 작업 완료! results.html을 확인하세요.")

def generate_html(results):
    html_content = """
    <html>
    <head>
        <title>WikiArt Classification Results</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; height: auto; }
            .yes { color: green; font-weight: bold; }
            .no { color: red; }
        </style>
    </head>
    <body>
        <h2>WikiArt Classification Report</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Human</th>
                <th>Animal</th>
                <th>Flower</th>
                <th>Reasoning</th>
            </tr>
    """
    
    for res in results:
        html_content += f"""
            <tr>
                <td><img src="{res['image_path']}"></td>
                <td class="{res['has_human']}">{res['has_human']}</td>
                <td class="{res['has_animal']}">{res['has_animal']}</td>
                <td class="{res['has_flower']}">{res['has_flower']}</td>
                <td>{res['reasoning']}</td>
            </tr>
        """
    
    html_content += "</table></body></html>"
    
    with open("results.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()