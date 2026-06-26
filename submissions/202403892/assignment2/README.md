# Assignment 2: WikiArt Image Classification

학번: 202403892 | 이름: Yunseo Lee

WikiArt 이미지 20개를 OpenRouter API로 분류하고 HTML 리포트를 생성하는 구현입니다.

## 사용 모델

```
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

추론 오버헤드를 줄이기 위해 `reasoning.effort: none`으로 설정했습니다.

## 실행 방법

```bash
pip install -r requirements.txt
```

`.env` 파일을 만들어 API 키를 설정합니다:

```
OPENROUTER_TOKEN=your_token_here
```

기본 실행 (20개 분류):

```bash
python classify_wikiart.py
```

옵션을 바꾸려면:

```bash
python classify_wikiart.py --limit 20 --max-size 320 --timeout 45 --delay 1.5
python classify_wikiart.py --model openai/gpt-4o-mini --limit 5
```

## 출력

```
results.html   # 이미지·분류 결과·근거를 보여주는 HTML 리포트
results.json   # 동일한 결과를 JSON으로 저장
images/        # 리사이즈된 썸네일 이미지
```

## Agentic Coding 주요 지시문

1. **구조 지시**: "OpenRouter API와 JSON Schema를 사용해서 WikiArt 이미지 20개를 분류하는 스크립트를 만들어줘. has_human, has_animal, has_flower를 yes/no로 분류하고, 판단 근거는 위치 표현(center, foreground 등)을 포함한 한 문장으로 작성하게 해줘. 결과는 HTML 파일 하나로 정리해줘."

2. **단순화 지시**: "멀티프로세싱 없이 순차적으로 처리하도록 해줘. .env 파일에서 OPENROUTER_TOKEN을 읽고, API 응답이 None이거나 비어있을 때 에러를 명확하게 처리하도록 해줘. 이미지는 상대경로로 저장해서 HTML 파일을 다른 곳에서도 열 수 있게 해줘."

3. **프롬프트 지시**: "모델 프롬프트에 has_human은 인물·초상·신체 부위 포함, has_animal은 조류·어류·곤충·신화 속 동물 포함, has_flower는 꽃 요소 포함이라는 기준을 명확하게 설명해줘. 그림 스타일이나 재질이 아니라 그려진 대상만 보고 판단하도록 명시해줘."

## 흥미로웠던 사례 3개

**1. 샘플 13 — 남성 초상화에서 라펠 꽃 발견**
> "A human figure is clearly visible in the center of the image, no animals are present, and a red flower is visible on the man's lapel."

모델이 초상화 속 인물의 옷깃에 달린 작은 빨간 꽃을 포착해 `has_flower: yes`로 분류했습니다. 단순히 "꽃이 없는 인물화"로 넘기기 쉬운 그림인데도 세밀하게 관찰했습니다.

**2. 샘플 7 — 달빛 풍경에서 불분명한 인물 처리**
> "the scene depicts a moonlit landscape with trees, clouds, and a distant figure that is too indistinct to confirm as human."

달빛 풍경화에 흐릿한 인물이 있었지만, 모델이 "확실하지 않다"고 판단해 `has_human: no`로 분류했습니다. 근거에서 직접 "too indistinct to confirm"이라고 설명한 점이 흥미롭습니다.

**3. 샘플 5 — 농촌 장면에서 복수 피사체 동시 인식**
> "Humans are clearly visible in the foreground and background, a black and white cow is visible in the lower right foreground, and no flowers are clearly visible."

인물 여러 명과 흑백 소를 동시에 구별해 `has_human: yes`, `has_animal: yes`, `has_flower: no`로 정확히 분류하고, 각 피사체의 위치까지 명시했습니다.

## 모델이 헷갈리거나 틀렸다고 생각되는 사례 2개

**1. 샘플 11 — evidence 필드 반복 생성 (hallucination)**
```
"depicted: depicted: depicted or or or or or or or or or or or or..."
```
레이블(`has_human: yes`, `has_animal: no`, `has_flower: no`)은 올바르게 나왔지만, evidence 필드에서 모델이 자기 생각을 정리하는 내부 텍스트를 그대로 출력하며 루프에 빠졌습니다. response-healing 플러그인이 JSON 형식은 살렸지만 내용 품질까지는 보장하지 못했습니다.

**2. 샘플 0, 15 — 응답 없음 (timeout 300초)**
두 샘플이 각각 300초 후 `'NoneType' object is not subscriptable` 에러로 실패했습니다. 모델이 응답 자체를 생성하지 못하고 빈 content를 반환한 것으로, 서버 부하나 이미지 처리 실패로 추정됩니다. 이 경우 재시도 로직이 있었다면 복구할 수 있었을 것입니다.
