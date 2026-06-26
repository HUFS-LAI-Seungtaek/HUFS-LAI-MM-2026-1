# Assignment 2 — WikiArt Image Classification

**Student:** Yeonjoo Yoo (202400220)

WikiArt 데이터셋의 그림 20장을 대상으로 세 가지 객체(인물, 동물, 꽃)의 존재 여부를 비전 언어 모델로 자동 분류

---

## 사용한 모델

**`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`** (OpenRouter 경유)

- NVIDIA의 멀티모달 추론 모델로, 텍스트·이미지·영상·오디오 입력 지원

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 토큰 설정

프로젝트 디렉토리에 `.env` 파일을 생성하고 OpenRouter 토큰을 입력합니다.

```
OPENROUTER_TOKEN=your_token_here
```

### 3. 분류 실행

```bash
# 기본 실행 (20개 샘플)
python classify_wikiart.py

# 샘플 수·모델 등 옵션 지정
python classify_wikiart.py --limit 20 --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free

# 이전 실행에서 실패한 샘플만 재시도
python classify_wikiart.py --resume
```

실행 결과로 `results.html`, `results.json`, `results.pdf` 세 파일이 생성됩니다.

---

## Agentic Coding 도구에 준 주요 지시문

### 1. JSON Schema를 이용한 구조화 출력 강제

모델이 자유 형식 텍스트 대신 미리 정의한 스키마에 맞는 JSON만 반환하도록 지시했습니다.

```
has_human, has_animal, has_flower 각각 "yes" / "no" 중 하나로만 대답하고,
evidence 필드에는 판단 근거를 한 문장으로 기술하라.
additionalProperties: false 로 스키마 외 필드를 금지한다.
```

이를 통해 파싱 오류 없이 결과를 바로 JSON으로 처리할 수 있었습니다.

### 2. 공간적 위치 정보를 포함한 근거(evidence) 생성 요구

모델이 단순히 yes/no만 반환하는 게 아니라, 판단의 근거를 함께 기술하도록 지시했습니다.

```
evidence에는 center, top-left, foreground, background 등
이미지 내 위치를 구체적으로 언급하여 한 문장으로 설명하라.
```

이를 통해 모델이 어떤 시각적 단서를 바탕으로 판단했는지 추적할 수 있었습니다.

---

## 결과에서 흥미로웠던 사례 3개

### 사례 1 — 옷깃의 꽃 장식까지 감지 (샘플 13)

```
has_human: yes / has_animal: no / has_flower: yes
evidence: A bearded man in a black coat sits centrally,
          with a red flower pinned to his left lapel.
```

실제 꽃이 아니라 인물 옷깃에 달린 작은 장식 꽃을 감지했습니다. 화면 전체를 차지하는 인물에 비해 꽃은 매우 작은 디테일임에도 위치(left lapel)까지 정확히 짚어냈습니다.

### 사례 2 — 배경 집 창문 화분의 꽃 발견 (샘플 5)

```
has_human: yes / has_animal: yes / has_flower: yes
evidence: Humans are visible in the foreground and mid-ground,
          a cow is in the lower right foreground, and flowers appear
          in the window box of the blue house on the right.
```

전경의 인물·소에 비해 훨씬 작은 배경 건물 창문에 놓인 화분 꽃까지 식별했습니다. 주의 대상이 아닌 배경 요소에서도 정의에 맞는 객체를 찾아낸 점이 인상적입니다.

### 사례 3 — 꽃만 있는 정물화 (샘플 6)

```
has_human: no / has_animal: no / has_flower: yes
evidence: A bouquet of flowers is prominently displayed in the center
          of the image, with additional flowers visible on the table
          in the foreground.
```

20개 샘플 중 인물도 동물도 없는 순수 정물화는 이 샘플이 유일했습니다. 다른 샘플 대부분에 인물이 등장하는 것과 대조적으로, 모델이 없는 객체를 있다고 환각하지 않고 정확히 꽃만 감지한 점이 흥미롭습니다.

---

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

### 사례 1 — 내부 추론 텍스트가 evidence 필드로 유출 (샘플 1)

```
has_human: yes / has_animal: yes / has_flower: yes
evidence: A human figure stands in the lower-right the message:
          The user wants to the user. Wait, no, the evidence must be
          a single sentence. Let's make it:
```

모델의 내부 chain-of-thought 추론 과정이 최종 출력 필드로 그대로 새어나왔습니다. 추론 모델(reasoning model)의 특성상 답변을 생성하기 전에 내부적으로 생각하는 과정이 있는데, 이 과정이 structured output과 분리되지 않은 것으로 보입니다. 분류 결과 자체(yes/yes/yes)는 맞을 수 있으나 evidence의 신뢰성이 없습니다.

### 사례 2 — 병사 갑옷 그림에서 동물 없음 판정 (샘플 11)

```
has_human: yes / has_animal: no / has_flower: no
evidence: The image shows soldiers in armor, indicating humans
          are present, while no animals or flowers are mentioned.
```

evidence에서 "no animals or flowers are **mentioned**"라는 표현은 이미지를 직접 묘사하는 것이 아니라 텍스트 언급 여부를 말하는 것처럼 읽혀, 모델이 시각 정보가 아닌 다른 단서에 의존했을 가능성을 시사합니다.
