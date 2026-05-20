# Assignment 2: Schema-Guided Image Classification on WikiArt

**Student:** 202403234 Jihyun Yang

## 사용한 모델

- **Primary:** `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (OpenRouter)
- **Fallback:** `google/gemini-2.0-flash-exp:free` (primary 실패 시 재시도)

## 실행 방법

1. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. `.env` 파일 생성 (제출하지 말 것)

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

4. 분류 실행

```bash
python classify_wikiart.py --count 20
```

5. 결과 확인

```bash
open results.html
```

## Agentic Coding 도구에 준 주요 지시문

1. `has_human`, `has_animal`, `has_flower`, `looks_ai_generated`, `reason`, `ai_reason` 필드를 가진 JSON Schema를 정의하고, OpenRouter API의 `response_format`으로 응답 형식을 강제하도록 작성하라. 프롬프트에는 각 필드의 정의를 명확히 설명하고, 위치 표현(center, foreground, top-left 등)을 포함한 근거를 작성하게 하라.

2. HuggingFace datasets HTTP API에서 이미지 URL을 가져와 로컬에 저장하고, base64로 인코딩해서 OpenRouter vision 모델에 전송하라. API 응답이 비어있을 경우 fallback 모델로 재시도하고, 그래도 실패하면 스킵 후 계속 진행하도록 에러 핸들링을 추가하라.

3. 모델이 "이 그림이 AI가 그린 것처럼 보이냐"를 판단하도록 프롬프트에 추가하라. 판단 근거로 texture consistency, anatomical accuracy, brushstroke irregularity 같은 시각적 단서를 사용하게 하라. 모든 이미지가 실제로는 역사적 인간 작품임을 명시하라.

## 결과에서 흥미로웠던 사례

1. **Sample 2**: 단일 이미지에서 천사(왼쪽 위), 목동들(오른쪽 아래), 소와 개까지 동시에 감지했다. `has_human`과 `has_animal`이 모두 yes로 나왔고, 복잡한 역사화에서 여러 카테고리를 함께 잡아낸 점이 인상적이었다.

2. **Sample 7**: 두꺼운 붓으로 표현된 꽃을 `has_flower: yes`로 판단했다. 일반적으로 ai가 만들어 낼 것같은 사진처럼 선명한 꽃이 아니어도 회화적 표현을 꽃으로 인식한 점이 흥미로웠다.

3. **전체 공통**: 분류에 성공한 모든 샘플에서 `looks_ai_generated: no`가 나왔다. 데이터가 수백 년 전 고전 회화들이라 붓터치나 불규칙한 질감 같은 인간적 특성이 뚜렷했고 헷갈릴수 있다고 생각했는데 모델이 AI 생성 이미지가 아님을 명확히 구분해낸 점이 흥미로웠다.

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례

1. **Sample 1**: 하늘에 작은 새 두 마리가 있다는 이유로 `has_animal: yes`로 분류했는데 실제로 이미지에서 새는 매우 작고 풍경의 일부에 해당해서 동물 분류 기준이 얼마나 엄격한지에 따라 결과가 달라질 수 있는 애매한 케이스였다.

2. **Sample 5, 6, 8, 9 등 다수**: Nemotron 모델이 응답을 반환하지 않았다... fallback인 Gemini도 404 에러로 실패했다. 코드 로직 문제가 아니라 무료 모델의 가용성 문제로 스킵이 발생했다. 실제 이미지를 보면 충분히 분류 가능한 작품들이라 모델 불안정성이 아쉬웠다. (일부 샘플은 코덱스 요금 한도를 초과해 추가 개선 없이 제출되었습니다!)
