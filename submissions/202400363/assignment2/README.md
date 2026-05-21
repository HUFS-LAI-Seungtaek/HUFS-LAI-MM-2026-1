# Assignment 2: WikiArt Image Classification with Schema-Guided Decoding

OpenRouter API로 `huggan/wikiart` 이미지 22개를 분류하고, 결과를 하나의 HTML 파일로 저장하는 구현입니다. 분류 근거는 자연어 문장으로 출력합니다.

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

**모델 특징:**
- NVIDIA의 경량형 멀티모달 모델 (30B parameters)
- 이미지와 텍스트 입력을 모두 처리 가능
- 빠른 추론 속도로 대량 처리에 적합
- 참고: 이 구현은 분류 속도를 위해 `reasoning.enabled`를 끄고 있습니다

다른 모델을 사용하려면 `--model` 옵션을 사용하세요.

## 주요 기능

- **Schema-Guided Decoding**: JSON Schema를 사용해서 모델 응답을 구조화된 형식으로 강제합니다
- **이미지 분류**: 각 이미지에 대해 human/animal/flower 포함 여부를 yes/no로 분류합니다
- **자연어 근거**: 각 분류에 대한 위치 정보를 포함한 자연어 설명을 생성합니다
- **병렬 처리**: `--workers` 옵션으로 여러 이미지를 동시에 처리합니다
- **타임아웃 처리**: 오래 걸리는 요청을 자동으로 중단하고 기록합니다
- **에러 추적**: 실패한 샘플도 결과에 포함해서 전체 프로세스 가시성을 제공합니다
- **HTML 리포트**: 분류 결과를 시각적인 HTML 파일로 생성합니다
- **메타데이터 보존**: 각 이미지의 artist, genre, style 정보를 유지합니다

## 실행 방법

먼저 의존성을 설치합니다:

```bash
pip install -r requirements.txt
```

기본 설정(22개 이미지)으로 실행합니다:

```bash
python classify_wikiart.py --limit 22
```

짧게 테스트하려면:

```bash
python classify_wikiart.py --limit 5
```

다른 이미지 개수로 실행하려면:

```bash
python classify_wikiart.py --limit 100 --workers 1
```

이미지 크기와 병렬 처리 수를 조정하려면:

```bash
python classify_wikiart.py --limit 22 --max-size 256 --workers 2 --request-timeout 10
```

**주의사항:**
- 요청 제한이 걸리면 `--workers 1`로 낮춰서 실행하세요
- 이미지가 너무 작아 분류가 불안정하면 `--max-size 512`로 올려보세요
- 샘플 하나가 너무 오래 걸리면 별도 프로세스를 종료하고 timeout으로 기록한 뒤 다음 샘플로 넘어갑니다

## 출력

```text
results.html      # 시각적 리포트 (이미지, 분류 결과, 자연어 근거 표시)
results.json      # JSON 형식의 원본 결과 (프로그래밍 분석용)
images/           # 22개 처리된 이미지 썸네일 (sample_000.jpg ~ sample_021.jpg)
```

**results.html:**
- 웹 브라우저에서 열 수 있는 대화형 테이블
- 각 이미지 썸네일, 메타데이터(artist, genre, style), 분류 결과, 근거 표시
- 성공한 샘플: 녹색(yes), 빨강색(no) 배경으로 시각적 강조

**results.json:**
- JSON 형식으로 모든 결과 데이터 저장
- 추가 분석, 시각화, 결과 비교에 사용
- 각 샘플의 index, metadata, labels, error, 처리 시간 포함

## 환경 설정

`.env` 파일에서 OPENROUTER_TOKEN을 설정합니다:

```
OPENROUTER_TOKEN=your_token_here
```

또는 환경 변수로 직접 설정할 수 있습니다:

```bash
export OPENROUTER_TOKEN=your_token_here
python classify_wikiart.py --limit 22
```

## Schema-Guided Decoding (JSON Schema 강제화)

이 구현은 OpenRouter API의 JSON Schema 기능을 사용해서 모델의 응답을 안정적으로 구조화합니다.

**Schema 정의:**
```json
{
  "type": "object",
  "properties": {
    "has_human": {"type": "string", "enum": ["yes", "no"]},
    "has_animal": {"type": "string", "enum": ["yes", "no"]},
    "has_flower": {"type": "string", "enum": ["yes", "no"]},
    "evidence": {
      "type": "string",
      "description": "A brief natural-language explanation with location information"
    }
  },
  "required": ["has_human", "has_animal", "has_flower", "evidence"],
  "additionalProperties": false
}
```

**이점:**
1. **응답 파싱 안정성**: 모든 응답이 정확히 정의된 형식을 따름
2. **필수 필드 보장**: `evidence` 등 모든 필드가 항상 존재
3. **값 제약**: `has_human`, `has_animal`, `has_flower`는 정확히 "yes" 또는 "no"만 가능
4. **자동 검증**: API 단에서 응답이 schema를 만족하지 않으면 거부

## 명령행 옵션

- `--limit`: 처리할 이미지 개수 (기본값: 100)
- `--sleep`: 요청 간 대기 시간 (기본값: 1.0초)
- `--output-html`: HTML 리포트 파일명 (기본값: results.html)
- `--output-json`: JSON 결과 파일명 (기본값: results.json)
- `--image-dir`: 이미지 저장 디렉토리 (기본값: images)
- `--max-size`: 이미지 최대 크기 (기본값: 256)
- `--workers`: 동시 작업 수 (기본값: 2)
- `--request-timeout`: 요청 타임아웃 (기본값: 10초)
- `--model`: 사용할 모델 (기본값: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free)

## Agentic Coding 프롬프트

이 프로젝트는 다음 두 가지 주요 지시문으로 AI 모델에게 구현을 요청했습니다:

### 지시문 1: 핵심 기능 정의
```
OpenRouter API를 사용해서 WikiArt 이미지 분류 스크립트를 만들어줘.
모델 응답은 JSON Schema로 고정하고, has_human/has_animal/has_flower를 yes/no로 분류하게 해줘.
프롬프트에는 사람, 동물, 꽃이 보이는지 판단하는 과제라는 점을 명확히 적어줘.
판단 근거는 위치 표현을 포함한 짧은 자연어 문장으로 저장하고, 
마지막에 단일 HTML 리포트를 생성하게 해줘.
```

### 지시문 2: 구현 최적화
```
코드를 간단하게 유지해줘. .env에서 OPENROUTER_TOKEN을 읽고,
Hugging Face datasets에서 huggan/wikiart 샘플을 가져와서 제한된 개수만 처리하게 해줘.
병렬 처리와 타임아웃 처리를 지원하고, 
실패한 샘플도 기록해서 HTML 리포트에 반영되게 해줘.
```

## 결과 분석

### 흥미로웠던 사례 3개

#### 1. Sample 5: 농업 장면의 복합 요소
```json
{
  "has_human": "yes",
  "has_animal": "yes",
  "has_flower": "no",
  "evidence": "Humans are visible in the foreground and mid-ground, a cow is in the lower-right foreground, and no flowers are clearly visible in the scene."
}
```
**특징:** 자연 풍경 속에서 인간, 동물, 그리고 주변 환경을 정확하게 식별했습니다. 위치 정보("foreground", "mid-ground", "lower-right")를 활용한 명확한 분류입니다.

#### 2. Sample 6: 정물화의 명확한 대상
```json
{
  "has_human": "no",
  "has_animal": "no",
  "has_flower": "yes",
  "evidence": "The artwork prominently features a bouquet of flowers in a vase, with no humans or animals visible."
}
```
**특징:** 꽃이 주제인 정물화를 정확하게 분류했습니다. 스타일이나 기법이 아닌 실제 객체 존재 여부에만 집중한 점이 좋습니다.

#### 3. Sample 18: 동물과의 상호작용
```json
{
  "has_human": "yes",
  "has_animal": "yes",
  "has_flower": "no",
  "evidence": "A person is riding a donkey in the center foreground, with other people nearby, but no flowers are visible."
}
```
**특징:** 복합적인 장면 속에서 인간과 동물의 상호작용을 포착했습니다. 장면의 중심 요소를 정확하게 파악합니다.

### 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

#### 1. Sample 13: 모순된 객체 인식
```json
{
  "has_human": "yes",
  "has_animal": "no",
  "has_flower": "yes",
  "evidence": "The image shows a seated adult male (human) wearing a black coat with a red flower on the lapel, indicating presence of a human and a flower; no animals are visible."
}
```
**문제점:** 설명에서 "red flower on the lapel"을 명확하게 언급했으므로 정확합니다. 하지만 초상화에서 꽃이 배경이 아닌 의류의 장식 요소인 경우, 모델이 이를 정확히 인식하는지 불확실한 부분입니다.

#### 2. Sample 16: 논리적 모순
```json
{
  "has_human": "yes",
  "has_animal": "yes",
  "has_flower": "no",
  "evidence": "The image depicts a human figure wearing a hooded garment, which is the only object type present; no animal or flower is visibly depicted."
}
```
**문제점:** 명백한 논리적 오류입니다. 설명에서 "which is the only object type present"라고 했으면서 동시에 "has_animal": "yes"라고 답하는 것은 모순입니다. 모델이 응답 형식과 자연어 설명 간의 일관성을 유지하지 못했습니다.
