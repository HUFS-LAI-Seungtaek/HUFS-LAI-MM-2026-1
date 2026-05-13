# Assignment 2: Schema-Guided WikiArt Classification

OpenRouter API와 JSON Schema 응답 형식을 사용해 `huggan/wikiart` 이미지를 분류하는 과제입니다.
현재 PR에는 빠른 검토를 위한 mock 결과 20개를 먼저 포함했습니다. 실제 WikiArt/OpenRouter 실행 경로는 같은 스크립트에 유지되어 있습니다.

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

분류 속도를 위해 OpenRouter 요청의 `reasoning`은 비활성화했습니다. 다른 vision-capable 모델을 쓰려면 `--model` 옵션으로 바꿀 수 있습니다.

## 실행 방법

의존성 설치:

```bash
pip install -r requirements.txt
```

mock 결과 재생성:

```bash
python classify_wikiart.py --mock --limit 20
```

실제 WikiArt 샘플 20개 분류:

```bash
python classify_wikiart.py --limit 20 --workers 1 --request-timeout 20
```

전체 실행:

```bash
python classify_wikiart.py --limit 100 --max-size 256 --workers 2 --request-timeout 20
```

`.env` 또는 환경 변수에 `OPENROUTER_TOKEN`이 필요합니다. `.env`는 제출하지 않습니다.

## 출력

```text
results.html
results.json
images/
```

`results.html`은 썸네일, metadata, `has_human`, `has_animal`, `has_flower`, 근거 문장을 한 표로 보여줍니다.
`results.json`은 같은 결과를 재분석하기 위한 원본 JSON입니다.

## Schema-Guided Decoding

모델 응답은 OpenRouter의 OpenAI-compatible API에 `response_format.type = json_schema`로 요청합니다. 스키마는 다음 네 필드를 강제합니다.

- `has_human`: `yes` 또는 `no`
- `has_animal`: `yes` 또는 `no`
- `has_flower`: `yes` 또는 `no`
- `evidence`: 위치 표현을 포함한 짧은 자연어 근거

## Agentic Coding Prompt

```text
OpenRouter API를 사용해서 WikiArt 이미지 분류 스크립트를 만들어줘.
모델 응답은 JSON Schema로 고정하고, has_human/has_animal/has_flower를 yes/no로 분류하게 해줘.
프롬프트에는 사람, 동물, 꽃이 보이는지 판단하는 과제라는 점을 명확히 적어줘.
판단 근거는 위치 표현을 포함한 짧은 자연어 문장으로 저장하고, 마지막에 단일 HTML 리포트를 생성하게 해줘.
```

```text
일단 PR을 빨리 열 수 있도록 mock 모드를 추가해줘.
mock 모드는 Hugging Face와 OpenRouter를 호출하지 않고 20개 샘플 이미지, JSON 결과, HTML 리포트를 생성해야 해.
실제 API 실행 경로는 나중에 그대로 사용할 수 있게 유지해줘.
```

## 흥미로웠던 사례

- `sample_006`: 사람, 동물, 꽃이 모두 있는 복합 케이스라 세 개 label이 동시에 `yes`가 되는지 확인하기 좋았습니다.
- `sample_007`: 아무 대상도 없는 배경 중심 이미지라 `no/no/no` 케이스를 확인할 수 있었습니다.
- `sample_013`: 동물과 꽃은 있지만 사람은 없는 조합이라 binary label이 독립적으로 저장되는지 보기 좋았습니다.

## 헷갈릴 수 있는 사례

- `sample_001`: mock 동물은 단순한 실루엣이라 실제 회화에서는 배경 장식이나 그림자로 오인될 수 있습니다.
- `sample_002`: 꽃 모양이 작고 단순해서 실제 모델에서는 색 덩어리나 장식 패턴으로 판단할 가능성이 있습니다.
