# Assignment 2: Schema-Guided WikiArt Image Classification

OpenRouter API와 JSON Schema 기반 응답 형식 고정을 사용해 `huggan/wikiart` 이미지 20개에서 사람, 동물, 꽃의 존재 여부를 분류했다. 결과는 `results.html`과 PDF 파일에 이미지, yes/no 라벨, 자연어 근거를 함께 정리했다.

## 제출 정보

- Student ID: `202403804`
- Name: `Jinhee Kim`
- PR title: `Assignment 2 by 202403804 (Jinhee Kim)`

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

OpenRouter 호출에는 `extra_body={"reasoning": {"enabled": True}}`를 사용했다. 이 모델은 첫 응답에서 `content` 없이 `reasoning_details`만 반환하는 경우가 있어서, 첫 호출의 reasoning state를 보존한 뒤 두 번째 호출에서 JSON Schema 형식의 최종 답변을 받도록 구현했다.

## 실행 방법

```bash
pip install -r requirements.txt
set OPENROUTER_TOKEN=your_openrouter_token
python classify_wikiart.py
```

중간에 API 호출이 끊기면 기존 `results.json`을 유지한 채 다음처럼 이어서 실행할 수 있다.

```bash
python classify_wikiart.py --resume
```

생성 파일은 다음과 같다.

```text
results.html
results.json
images/
WikiArt Classification Results.pdf
```

API token이 들어간 `.env` 파일은 제출하지 않는다.

## Agentic Coding 지시문

```text
OpenRouter API를 사용해서 WikiArt 이미지 분류 스크립트를 만들어줘.
모델 응답은 JSON Schema로 고정하고, has_human/has_animal/has_flower를 yes/no로 분류하게 해줘.
프롬프트에는 사람, 동물, 꽃이 보이는지 판단하는 과제라는 점을 명확히 적어줘.
판단 근거에는 center, top-left, foreground 같은 위치 표현을 가능한 한 포함하고, 마지막에 단일 HTML 리포트를 생성하게 해줘.
```

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free 모델은 reasoning_details를 보존해서 이어서 호출해야 하므로,
첫 호출에서 reasoning을 받은 뒤 두 번째 호출에서 JSON Schema 최종 결과를 받는 방식으로 고쳐줘.
API 호출이 오래 걸리거나 timeout이 나도 결과를 잃지 않도록 resume 기능과 샘플별 저장을 넣어줘.
```

## 흥미로웠던 사례

- Sample 0: 풍경처럼 보이는 이미지에서 top-left의 작은 새 두 마리를 animal로 잡아냈다.
- Sample 13: 인물 초상에서 옷깃의 빨간 꽃을 flower로 분류했고, 사람과 꽃을 동시에 yes로 판단했다.
- Sample 15: human, animal, flower가 모두 yes인 복합 사례로, top-left의 새와 center-left의 꽃을 함께 근거로 들었다.

## 헷갈렸거나 틀렸을 수 있는 사례

- Sample 1: 천사형 인물을 human으로 보고, lower-right에 animal이 있다고 판단했는데 이미지 표현이 작거나 상징적이면 사람/동물 기준이 애매할 수 있다.
- Sample 14: background right의 새를 animal로 분류했지만, 배경의 작은 형상이므로 사람이 직접 확인할 필요가 있다.
