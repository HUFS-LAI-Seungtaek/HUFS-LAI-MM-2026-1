# Assignment 2 Reference

OpenRouter API로 `huggan/wikiart` 이미지 100개를 분류하고, 결과를 하나의 HTML 파일로 저장하는 참고 구현입니다. 분류 근거는 자연어 문장으로 출력합니다.

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

참고: 이 reference 코드는 분류 속도를 위해 `reasoning.enabled`를 켜지 않습니다. 모델을 바꾸고 싶다면 `--model` 옵션을 사용하세요.

## 실행 방법

```bash
pip install -r requirements.txt
python classify_wikiart.py --limit 100
```

먼저 짧게 테스트하려면:

```bash
python classify_wikiart.py --limit 5
```

이미지 크기와 병렬 처리 수를 조정하려면:

```bash
python classify_wikiart.py --limit 100 --max-size 256 --workers 2 --request-timeout 10
```

요청 제한이 걸리면 `--workers 1`로 낮춰서 실행하세요.
이미지가 너무 작아 분류가 불안정하면 `--max-size 512`로 올려보세요.
샘플 하나가 너무 오래 걸리면 timeout으로 처리하고 다음 샘플로 넘어갑니다.

다른 모델을 사용하려면:

```bash
python classify_wikiart.py --limit 5 --model openai/gpt-4o-mini
```

## 출력

```text
results.html
results.json
images/
```

`results.html`은 `images/` 폴더의 썸네일, 모델 분류 결과, 자연어 근거를 보여주는 HTML 파일입니다.
`results.json`은 같은 결과를 재분석하기 쉽게 저장한 원본 결과 파일입니다.

## Agentic Coding Prompt 예시

```text
OpenRouter API를 사용해서 WikiArt 이미지 분류 스크립트를 만들어줘.
모델 응답은 JSON Schema로 고정하고, has_human/has_animal/has_flower를 yes/no로 분류하게 해줘.
프롬프트에는 사람, 동물, 꽃이 보이는지 판단하는 과제라는 점을 명확히 적어줘.
판단 근거는 위치 표현을 포함한 짧은 자연어 문장으로 저장하고, 마지막에 단일 HTML 리포트를 생성하게 해줘.
```

```text
코드를 간단하게 유지해줘. .env에서 OPENROUTER_TOKEN을 읽고,
Hugging Face datasets에서 huggan/wikiart 샘플을 가져와서 100개만 처리하게 해줘.
```
