# Assignment 2
- Language & AI 융합전공 202402050 최재원**

---

## 사용한 모델

**nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free**
> 할당량을 모두 사용하여 `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`를 서빙해 사용함

---

## 실행 방법

### 1. 환경 설정
```bash
conda create -n mm-ass2 python=3.11
conda activate mm-ass2
pip install -r requirements.txt
```

### 2. API 키 설정
`.env`에 OpenRouter API 토큰 설정:
```
OPENROUTER_TOKEN=<token>
```

### 3. 코드 실행
```bash
python classify_wikiart.py
```

---

## Agentic Coding 도구에 준 주요 지시문

### 지시문 1: 가능한 모든 내용을 포함해 지시

```
HF datasets `huggan/wikiart`의 초기 100개를 선택해 human, animal, flower의 세 가지가 존재하는지 분류해야 해
has_human, has_animal, has_flower에 대해 `yes`나 `no`로만 분류 및 `reason`에 판단 근거를 제시하도록 schema-guided decoding을 활용해야 해
이를 위한 프롬프트에 `center`, `foreground`, `background` 등의 위치 표현을 포함하도록 해주고, `has_human`, `has_animal`, `has_flower`의 의미를 명확히 설명하도록 해줘
이를 위한 MLLM으로, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`를 OpenRouter API로 사용할 거고, API 키는 .env에 `OPENROUTER_TOKEN`으로 있어
conda 환경으로 `mm-ass2`를 만들어 사용하고, requirements.txt에 사용된 패키지들을 버전과 함께 명시해줘
샘플은 20개로 제한해 분류하고, 결과를 `results.html`에 시각화해주고, assets는 `images/` 폴더에 저장해주는 파이썬 스크립트를 `classify_wikiart.py`에 작성해주면 돼
OpenRouter 구현은 아래를 참고해
(ref: https://openrouter.ai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free/api - OpenAI SDK)
```
> 과제 내용 이해 및 기본 구현을 위함

### 지시문 2: 불필요한 구조 제거

```
병렬 처리는 필요 없고, JSON 형태로 반환하지 못한 경우나 Timeout 등에 대해서는 다시 시도하도록 해줘
```
> 멀티쓰레딩으로 구현된 복잡한 구조를 덜어내고, 어쨌든 모든 샘플에 대해 획득할 수 있도록 수정하고자

---

## 결과에서 흥미로웠던 사례 3개

### 사례 1: TBD

### 사례 2: TBD

### 사례 3: TBD

---

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

### 문제 사례 1: TBD

### 문제 사례 2: TBD
