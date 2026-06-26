# Assignment 2: Schema-Guided Image Classification on WikiArt

**Student:** Jisoo Jang (202400420)

## 사용한 모델

`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` via OpenRouter API

## 실행 방법

```bash
pip install -r requirements.txt

# .env 파일에 API 키 설정
echo "OPENROUTER_TOKEN=your_key_here" > .env

# 분류 실행 (기본 20개 샘플)
python classify_wikiart.py

# 옵션 예시
python classify_wikiart.py --limit 20 --sleep 1.5 --timeout 90
```

결과 파일: `results.html`, `results.json`, `images/`

## Agentic Coding 도구에 준 주요 지시문

```
과제를 해야됨. 이 README.md 파일을 읽고 schema-guided decoding을 활용해서 openrouter.ai api를 갖고 이미지 분류 작업을 해줘. 
오픈 라우터 key는: sk-or-v1-****
```

## 흥미로웠던 사례 3개

### Sample 5 - 복잡한 화면에서 디테일 감지
모델이 화면 중앙에 존재하는 다수의 사람들(human), 오른쪽 아래의 젖소(animal), 건물 창틀 위 꽃 화분(flower)을 각각 정확한 위치와 함께 식별했다. 화분은 인간 시야에서도 자칫하면 놓치기 쉬운만큼 작게 표현되어있고, 그림이 전체적으로 복잡하게 구성되어있는데도 이런 복잡한 화면에서 작은 소품까지 놓치지 않은 점이 흥미롭다.

### Sample 7 - 어두운 화면에서 동물 형상 감지
어둡고 흐린 화면에서 모델이 동물의 형상을 정확하게 식별하였다. 인간 시야에서도 유심히 보지 않으면 화면 아래에 있는 형상이 무엇인지 정확하게 구별이 안될법하지만, "sheep or cattle"이라는 evidence와 함께 정확하게 구분해내는 모습이다.

### Sample 13 - 부토니에 감지
정장을 입은 남성의 옷깃에 꽂힌 부토니에을 보고 꽃으로 판정하며 "a red flower is visible on the man's lapel"이라고 evidence 남겼다. 흔히 꽃은 화분이나 꽃병 또는 자연물의 형태로 표현되곤 하지만, 해당 사례처럼 옷에 장식품으로 쓰인 꽃도 정확하게 구분해낸 것을 볼 수 있다.

## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개

### Sample 9 — evidence 필드 누락
results.html 화면에서 evidence가 누락되어 실제 json 파일을 확인해보았다. `has_human: yes`, `has_animal: no`, `has_flower: no`는 반환했지만 evidence에서는 응답에서 빠졌다. 불완전한 응답에도 불구하고 response-healing 플러그인이 JSON 구조는 복구했지만 evidence는 누락상태 그대로 출력되었다.

### Sample 3 - 인상주의 그림에서의 모호성
모델이 flower는 없고 bushes and trees만 있다고 evidence를 서술하였다. 하지만 인상주의 그림에서 흔히 꽃을 포함한 식물들은 명확한 형태 없이 색으로만 표현되며, 인간의 눈에서는 해당 색 덩어리에 초록색 외의 붉은색, 흰색 등이 섞여있을 경우 그 부분을 꽃으로 인식하곤 한다. 이러한 부분을 모델에게 명시적으로 알려주지 않아서 형태 없이 색만으로는 flower가 없다고 판별한 것이 흥미롭다. 