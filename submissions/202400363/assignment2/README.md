# Assignment 2: WikiArt Image Classification with Schema-Guided Decoding

OpenRouter API로 `huggan/wikiart` 이미지 100개를 분류하고, 결과를 하나의 HTML 파일로 저장하는 구현입니다. 분류 근거는 자연어 문장으로 출력합니다.

## 사용 모델

```text
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

참고: 이 구현은 분류 속도를 위해 `reasoning.enabled`를 끄고 있습니다. 모델을 바꾸고 싶다면 `--model` 옵션을 사용하세요.

## 주요 기능

- **Schema-Guided Decoding**: JSON Schema를 사용해서 모델 응답을 구조화된 형식으로 강제합니다
- **이미지 분류**: 각 이미지에 대해 human/animal/flower 포함 여부를 yes/no로 분류합니다
- **자연어 근거**: 각 분류에 대한 위치 정보를 포함한 자연어 설명을 생성합니다
- **병렬 처리**: 여러 이미지를 동시에 처리합니다
- **HTML 리포트**: 분류 결과를 시각적인 HTML 파일로 생성합니다

## 실행 방법

먼저 의존성을 설치합니다:

```bash
pip install -r requirements.txt
```

기본 설정(100개 이미지)으로 실행합니다:

```bash
python classify_wikiart.py --limit 100
```

짧게 테스트하려면:

```bash
python classify_wikiart.py --limit 5
```

이미지 크기와 병렬 처리 수를 조정하려면:

```bash
python classify_wikiart.py --limit 100 --max-size 256 --workers 2 --request-timeout 10
```

요청 제한이 걸리면 `--workers 1`로 낮춰서 실행하세요.
이미지가 너무 작아 분류가 불안정하면 `--max-size 512`로 올려보세요.
샘플 하나가 너무 오래 걸리면 별도 프로세스를 종료하고 timeout으로 기록한 뒤 다음 샘플로 넘어갑니다.

다른 모델을 사용하려면:

```bash
python classify_wikiart.py --limit 5 --model openai/gpt-4o-mini
```

## 출력

```text
results.html      # 시각적 리포트
results.json      # JSON 형식의 원본 결과
images/           # 처리된 이미지 썸네일
```

`results.html`은 `images/` 폴더의 썸네일, 모델 분류 결과, 자연어 근거를 보여주는 HTML 파일입니다.
`results.json`은 같은 결과를 재분석하기 쉽게 저장한 원본 결과 파일입니다.

## 환경 설정

`.env` 파일에서 OPENROUTER_TOKEN을 설정합니다:

```
OPENROUTER_TOKEN=your_token_here
```

또는 환경 변수로 직접 설정할 수 있습니다:

```bash
export OPENROUTER_TOKEN=your_token_here
python classify_wikiart.py --limit 100
```

## Schema-Guided Decoding

이 구현은 OpenRouter API의 JSON Schema 기능을 사용합니다. 모델은 다음 형식으로 응답하도록 강제됩니다:

```json
{
  "has_human": "yes" | "no",
  "has_animal": "yes" | "no",
  "has_flower": "yes" | "no",
  "evidence": "A brief natural-language explanation with location information"
}
```

이렇게 하면 응답을 안정적으로 파싱할 수 있고, 모델이 항상 필요한 정보를 제공하도록 보장합니다.

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
