import os
import json
import base64
import argparse
import requests
import time
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

def encode_image(image):
    """이미지를 Base64 인코딩 문자열로 변환합니다."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_image(base64_image, model, timeout):
    """OpenRouter API를 호출하여 이미지를 분류하고, 실패 시 재시도합니다."""
    token = os.getenv("OPENROUTER_TOKEN")
    if not token:
        return {"error": "OPENROUTER_TOKEN is not set in .env"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # JSON Schema 고정
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "wikiart_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "has_human": {"type": "string", "enum": ["yes", "no"]},
                    "has_animal": {"type": "string", "enum": ["yes", "no"]},
                    "has_flower": {"type": "string", "enum": ["yes", "no"]},
                    "reasoning": {
                        "type": "string",
                        "description": "판단 근거. 반드시 한국어로만 작성해야 하며, 중앙, 좌측 상단, 우측 하단 등의 구체적인 위치 표현을 포함하세요."
                    }
                },
                "required": ["has_human", "has_animal", "has_flower", "reasoning"],
                "additionalProperties": False
            }
        }
    }

    # 사람, 동물, 꽃 판단 과제 명시 및 위치 표현을 포함한 판단 근거 요청
    prompt_text = (
        "이 과제는 주어진 이미지에 사람(human), 동물(animal), 꽃(flower)이 "
        "보이는지 판단하는 과제입니다. 각 항목에 대해 'yes' 또는 'no'로 분류해 주세요. "
        "중요: 판단 근거(reasoning)는 반드시 '한국어(Korean)'로만 작성해 주세요. 영어, 아랍어 등 다른 언어는 절대 섞어 쓰지 마세요. "
        "또한 '중앙(center)', '좌측 상단(top-left)', '우측 하단(bottom-right)'과 같은 구체적인 위치 표현을 포함하여 판단 근거를 짧게 작성해 주세요."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "response_format": schema
    }
    # 참고: reasoning.enabled는 기본 설정에 맡기며 명시적으로 켜지 않습니다.

    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if res.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"  [Rate Limit 429] 서버 요청 제한. {wait}초 대기 후 재시도 (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
                
            res.raise_for_status()
            data = res.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content')
            
            if content is None:
                raise ValueError("Model returned empty or None content")
                
            return json.loads(content)
        except requests.exceptions.Timeout:
            wait = 2 * (attempt + 1)
            if attempt == max_retries - 1:
                return {"error": "timeout"}
            print(f"  [Timeout] 요청 시간 초과. {wait}초 대기 후 재시도 (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt == max_retries - 1:
                return {"error": str(e)}
            print(f"  [Error] 요청 중 오류 발생. {wait}초 대기 후 재시도 (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
    return {"error": "모든 재시도에 실패했습니다."}

def main():
    parser = argparse.ArgumentParser(description="Classify WikiArt images using OpenRouter API")
    parser.add_argument("--limit", type=int, default=20, help="처리할 이미지 수")
    parser.add_argument("--max-size", type=int, default=256, help="이미지 최대 크기 제한")
    parser.add_argument("--sleep", type=float, default=1.0, help="API 요청 간 대기 시간(초)")
    parser.add_argument("--request-timeout", type=int, default=20, help="API 요청 타임아웃(초)")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", help="사용할 모델명")
    args = parser.parse_args()

    load_dotenv()
    
    os.makedirs("images", exist_ok=True)
    
    print("Loading dataset...")
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    results_file = "results.json"
    results = []
    
    # 1. 기존에 저장된 결과(로그)가 있다면 불러와서 이어서 진행합니다.
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"기존에 완료된 {len(results)}개의 결과를 불러왔습니다. 이어서 진행합니다...")
        except Exception as e:
            print(f"기존 결과를 불러오는 중 오류 발생 (무시하고 새로 시작합니다): {e}")
            
    processed_ids = {r["id"] for r in results}
    
    print(f"Starting sequential classification for {args.limit} images...")
    for idx, item in enumerate(dataset):
        if idx >= args.limit:
            break
            
        if idx in processed_ids:
            print(f"[{idx+1}/{args.limit}] Skipping image {idx} (이미 처리됨)...")
            continue
            
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.thumbnail((args.max_size, args.max_size))
        image_path = f"images/image_{idx}.jpg"
        image.save(image_path, format="JPEG")
        
        b64_img = encode_image(image)
        
        print(f"[{idx+1}/{args.limit}] Classifying image {idx}...")
        res = classify_image(b64_img, args.model, args.request_timeout)
        
        result_entry = {"id": idx, "image_path": image_path, "b64_img": b64_img}
        if "error" in res:
            result_entry["error"] = res["error"]
            print(f"  -> Error: {res['error']}")
        else:
            result_entry.update(res)
            print(f"  -> Success | Human: {res.get('has_human')} | Animal: {res.get('has_animal')} | Flower: {res.get('has_flower')}")
        
        results.append(result_entry)
        
        # 2. 중간에 멈춰도 데이터가 날아가지 않도록 매 작업마다 JSON에 즉시 저장 (체크포인트)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 마지막 이미지가 아니면 API Rate Limit 방지를 위해 잠시 대기
        if idx < args.limit - 1:
            time.sleep(args.sleep)

    # id 기준으로 오름차순 정렬
    results.sort(key=lambda x: x["id"])
    
    # 모든 작업 완료 후 최종 JSON 저장
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 결과를 직관적으로 볼 수 있는 단일 HTML 리포트 생성
    print("Generating HTML report...")
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>WikiArt Classification Report</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ccc; padding: 10px; text-align: left; }",
        "th { background-color: #eee; }",
        "img { max-width: 100%; height: auto; border-radius: 4px; }",
        ".yes { color: blue; font-weight: bold; }",
        ".no { color: red; }",
        ".error { color: orange; font-weight: bold; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>WikiArt Image Classification Results</h1>",
        "<table>",
        "<tr><th>ID</th><th>Thumbnail</th><th>Human</th><th>Animal</th><th>Flower</th><th>Reasoning / Error</th></tr>"
    ]
    
    for r in results:
        html_lines.append("<tr>")
        html_lines.append(f"<td>{r['id']}</td>")
        
        # b64_img가 있으면 이미지를 직접 삽입하고, 없으면 기존 경로를 사용 (하위 호환성)
        img_src = f"data:image/jpeg;base64,{r['b64_img']}" if 'b64_img' in r else r['image_path']
        html_lines.append(f"<td><img src='{img_src}' alt='Image {r['id']}'></td>")
        
        if "error" in r:
            html_lines.append(f"<td colspan='4' class='error'>Error / Timeout: {r['error']}</td>")
        else:
            h_class = "yes" if r.get("has_human") == "yes" else "no"
            a_class = "yes" if r.get("has_animal") == "yes" else "no"
            f_class = "yes" if r.get("has_flower") == "yes" else "no"
            
            html_lines.append(f"<td class='{h_class}'>{r.get('has_human', '-')}</td>")
            html_lines.append(f"<td class='{a_class}'>{r.get('has_animal', '-')}</td>")
            html_lines.append(f"<td class='{f_class}'>{r.get('has_flower', '-')}</td>")
            html_lines.append(f"<td>{r.get('reasoning', '-')}</td>")
            
        html_lines.append("</tr>")
        
    html_lines.append("</table></body></html>")
    
    with open("results.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
        
    print("Done! Check results.json and results.html.")

if __name__ == "__main__":
    main()