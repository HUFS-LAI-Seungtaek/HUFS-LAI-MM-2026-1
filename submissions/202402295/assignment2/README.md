# Assignment 2: Schema-Guided Image Classification on WikiArt

## 사용한 모델

- OpenRouter model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`

## 실행 방법

1. 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

2. 로컬 전용 `.env` 파일을 만듭니다. 이 파일은 제출하지 않습니다.

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

3. WikiArt 이미지 20개를 내려받고 분류합니다.

```bash
python classify_wikiart.py --download --limit 20
```

Nemotron 무료 모델은 이미지 입력에서 응답이 느릴 수 있어, 이 구현은 이미지를 96px 썸네일로 줄이고 `max_tokens`를 제한해 안정적으로 처리합니다.

4. 이미 `images/` 폴더에 이미지가 있으면 다운로드 없이 실행할 수 있습니다.

```bash
python classify_wikiart.py --limit 20
```

실행이 끝나면 `results.html`과 `results.json`이 생성되고, 이 README의 사례 분석 섹션도 실행 결과를 바탕으로 자동 업데이트됩니다.

## Agentic coding 도구에 준 주요 지시문

1. OpenRouter API를 사용해서 WikiArt 이미지를 분류하고, 모델 응답은 JSON Schema로 고정해 `has_human`, `has_animal`, `has_flower`를 `yes`/`no`로만 반환하게 해달라고 지시했다.
2. 프롬프트에는 사람, 동물, 꽃이 보이는지 판단하는 과제라는 점을 명확히 적고, 판단 근거는 위치 표현을 포함한 짧은 자연어 문장으로 저장하게 해달라고 지시했다.
3. 과제 제출 안내사항에 맞춰 `README.md`, 실행 코드, `results.html`, `images/`, `requirements.txt` 구조를 만들고, API 키는 `.env`로 분리하되 제출하지 않게 해달라고 지시했다.

## 결과에서 흥미로웠던 사례 3개

1. 이미지 #6 (`sample_005.jpg`): human=yes, animal=yes, flower=no. Several people stand in the foreground and a cow-like animal appears near the right side, but no distinct flowers are visible.
2. 이미지 #12 (`sample_011.jpg`): human=yes, animal=yes, flower=no. A crowd of people and horse-like figures appears in the lower foreground near the fortress, with no flowers visible.
3. 이미지 #13 (`sample_012.jpg`): human=yes, animal=yes, flower=no. A tiny rider and horse-like form appear in the lower foreground against the mountain background, with no flowers present.

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

1. 이미지 #1 (`sample_000.jpg`): human=no, animal=no, flower=no. The foreground and background show a tree-lined landscape with no visible humans, animals, or flowers.
2. 이미지 #8 (`sample_007.jpg`): human=no, animal=no, flower=no. The dark landscape fills the center and background, with no discernible human, animal, or flower forms.

## 제출 전 확인

- `.env` 파일은 API 토큰을 포함하므로 제출하지 않습니다.
- `results.html`과 `results.json`은 `python classify_wikiart.py --download --limit 20` 실행 후 생성된 파일로 제출합니다.
- PR 제목 형식:

```text
Assignment 2 by 202402295 (Yuheon Sung)
```
