# Assignment 2: Schema-Guided Image Classification on WikiArt

OpenRouter API와 agentic coding 도구(Codex 또는 Claude Code)를 사용해 WikiArt 이미지를 분류하세요.

## 목표

[`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) 데이터셋에서 이미지 100개를 선택하고, 각 이미지에 대해 다음 값을 분류합니다.

- `has_human`: `yes` / `no`
- `has_animal`: `yes` / `no`
- `has_flower`: `yes` / `no`

모델이 그렇게 판단한 근거를 자연어로 짧게 작성하세요. 가능하면 `center`, `top-left`, `foreground` 같은 위치 표현을 포함하세요.

## 해야 할 일

- Codex 또는 Claude Code로 코드 작성
- OpenRouter API 사용
- JSON Schema로 모델 응답 형식 고정
- 프롬프트에 `has_human`, `has_animal`, `has_flower`의 의미를 명확히 설명
- 샘플 20개 분류
- 결과를 하나의 HTML 파일로 정리

## 제출 위치

본인 폴더를 만들어 제출하세요.

```text
submissions/{학번}/assignment2/
```

예시:

```text
submissions/2025122/assignment2/
```

## 제출 파일

- `README.md`
- 실행 코드: `classify_wikiart.py`
- 결과 파일: `results.html`
- 이미지 폴더: `images/`
- 필요한 경우 `requirements.txt`

API token이 들어간 `.env` 파일은 제출하지 마세요.

## README에 포함할 내용

- 사용한 모델
- 실행 방법
- agentic coding 도구에 준 주요 지시문 2개 이상
- 결과에서 흥미로웠던 사례 3개
- 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

## PR 제목

```text
Assignment 2 by {학번} ({영어 이름})
```

예시:

```text
Assignment 2 by 2025122 (Seungtaek Choi)
```

## 참고 구현

`submissions/2025122/assignment2/` 디렉토리에 참고 구현이 있습니다. 그대로 복사하기보다는 구조를 이해한 뒤 본인 방식으로 수정해서 제출하세요.
