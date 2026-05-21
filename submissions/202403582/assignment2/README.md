# Assignment 2: WikiArt Schema-Guided Classification

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

OpenRouter Chat Completions API를 사용했고, `response_format.type=json_schema`로 `has_human`, `has_animal`, `has_flower`, `evidence` 필드가 항상 나오도록 고정했다.

## 실행 방법

이 폴더 또는 저장소 루트에 `.env` 파일을 만들고 API 토큰을 넣는다.

```bash
OPENROUTER_TOKEN=your_openrouter_token_here
```

필요한 패키지를 설치한 뒤 실행한다.

```bash
cd submissions/202403582/assignment2
pip install -r requirements.txt
python classify_wikiart.py
```

기본 실행은 `huggan/wikiart` train split에서 이미지 20개를 streaming 방식으로 가져와 분류한다. 실행 결과는 아래 파일로 저장된다.

```text
results.html
results.json
WikiArt Classification Results.pdf
images/
```

테스트용으로 더 적은 이미지부터 실행하려면 다음처럼 한다.

```bash
python classify_wikiart.py --limit 3
```

이미 성공한 행은 유지하고 실패한 행만 다시 시도하려면 다음처럼 실행한다.

```bash
python classify_wikiart.py --limit 20 --resume --sleep 4 --request-timeout 60
```

OpenRouter 무료 모델 한도가 걸릴 수 있으므로, rate limit이 발생하면 reset 이후 위 명령으로 이어서 실행한다.
실행이 끝나면 `results.html`, `results.json`, `WikiArt Classification Results.pdf`가 함께 갱신된다.

## Agentic coding 도구에 준 주요 지시문

```text
assignments/assignment2/README.md를 읽고 과제 내용을 파악한 뒤,
submissions/202403582/assignment2/ 폴더 안에 과제물을 만들어줘.
```

```text
huggan/wikiart 데이터셋에서 이미지 20개를 스트리밍으로 가져와서
OpenRouter API (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free 모델)를 사용해
schema-guided decoding으로 이미지 분류를 해줘.
```

```text
분류 항목은 has_human, has_animal, has_flower (yes/no)이고,
판단 근거는 center, top-left, foreground 같은 위치 표현을 포함한 한 문장으로 작성해줘.
```

## 결과에서 흥미로웠던 사례 3개

1. `wikiart_000`에서 모델은 사람과 꽃은 없지만 top-left 하늘과 물가의 새를 근거로 `has_animal=yes`라고 판단했다.
2. `wikiart_001`에서 중앙의 천사형 인물과 lower-right의 양을 함께 찾아 `has_human=yes`, `has_animal=yes`로 분류했다.
3. `wikiart_003`에서 중앙 인물은 `has_human=yes`, top-left foliage의 꽃은 `has_flower=yes`라고 위치 근거를 포함해 설명했다.

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

1. `wikiart_003`의 꽃 판단은 배경 식물과 붓질이 섞여 있어 실제 꽃인지 장식적 색면인지 애매할 수 있다.
2. `wikiart_004`는 보트와 해변의 인물이 매우 작아 `has_human=yes` 근거가 이미지 크기나 압축 정도에 따라 흔들릴 수 있다.

## 실행 메모

현재 저장된 실행 결과는 20개 이미지 모두 OpenRouter API 분류가 완료된 상태다. `WikiArt Classification Results.pdf`는 GitHub PR 화면에서 바로 확인할 수 있도록 카드형 레이아웃으로 생성했다.
