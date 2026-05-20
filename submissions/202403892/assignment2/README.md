# WikiArt Image Classification using Schema-Guided Decoding

## 프로젝트 개요

이 프로젝트는 OpenRouter API와 Nvidia Nemotron 3 모델을 사용하여 WikiArt 데이터셋의 이미지를 분류합니다. **Schema-Guided Decoding**을 활용하여 모델의 응답 형식을 JSON Schema로 엄격하게 제한하고, 일관된 구조의 분류 결과를 보장합니다.

### 주요 기술
- **모델**: Nvidia Nemotron 3 Nano Omni (무료) (OpenRouter API)
- **기법**: Schema-Guided Decoding (JSON Schema 기반 구조화된 출력)
- **분류 항목**: 
  - `has_human`: 인물 포함 여부
  - `has_animal`: 동물 포함 여부
  - `has_flower`: 꽃 포함 여부
  - `reasoning`: 분류 근거 (위치 표현 포함)

## 사용한 모델

- **모델명**: Nvidia Nemotron 3 Nano Omni 30B A3B Reasoning
- **API 제공자**: OpenRouter (https://openrouter.ai/) - 무료 버전
- **Vision 능력**: 이미지 내용 분석 및 세부 요소 감지
- **추론 능력**: Chain-of-thought 추론으로 정확한 분류 근거 생성
- **출력 형식**: JSON Schema 기반 구조화된 응답

## 실행 방법

### 1. 환경 설정

```bash
# 저장소 클론 또는 다운로드
cd assignment2

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 생성하고 OpenRouter API 키를 추가합니다:

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 후 API 키 입력
OPENROUTER_API_KEY=your_actual_api_key_here
```

OpenRouter API 키는 https://openrouter.ai/ 에서 가져올 수 있습니다.

### 3. 이미지 분류 실행

```bash
# 주 실행 스크립트
python classify_wikiart.py
```

이 스크립트는:
1. Hugging Face에서 WikiArt 데이터셋 로드
2. 20개 이미지 다운로드 (images/ 폴더)
3. Claude Vision 모델로 각 이미지 분류
4. 결과를 JSON 및 HTML 형식으로 저장

### 4. 샘플 결과 생성 (API 없이)

```bash
python generate_sample_results.py
```

### 결과 확인

- **HTML 리포트**: `results.html` (브라우저로 열기)
- **JSON 데이터**: `results.json` (프로그래밍 처리용)
- **이미지 폴더**: `images/` (다운로드된 원본 이미지)

## Agentic Coding 도구 지시문

### 지시문 1: Schema-Guided Decoding 구현

```
"Claude의 JSON Schema 기반 응답 구조화 기능을 활용하여, 
모델의 출력을 엄격하게 제한하는 구조화된 응답 스키마를 설계했습니다.

구체적으로:
- JSON Schema를 정의하여 has_human, has_animal, has_flower, reasoning 필드를 명시
- response_format: { type: 'json_schema', json_schema: {..., strict: true } } 
  설정으로 응답이 항상 스키마를 준수하도록 강제
- 이를 통해 파싱 오류를 0%로 만들고 일관된 데이터 구조 보장

이는 대규모 자동화 파이프라인에서 매우 중요한 기능입니다."
```

### 지시문 2: 멀티모달 프롬프팅과 위치 표현

```
"이미지 분류 시 모델이 단순히 'yes/no'만 반환하는 것을 피하기 위해,
자연어 추론(reasoning)을 필수 필드로 포함시켰습니다.

프롬프트에서:
- '분류 근거를 자연어로 작성하세요'
- '위치 표현(center, top-left, foreground, background 등)을 포함하세요'
라고 명시적으로 지시

이를 통해:
1. 모델의 의사결정 과정을 추적 가능하게 함
2. 오류 패턴을 분석할 수 있는 추론 기록 제공
3. 사용자에게 분류의 신뢰도를 평가하는 근거 제시"
```

### 지시문 3: 배치 처리와 에러 핸들링

```
"대량의 이미지를 처리할 때 안정성을 위해:

1. 각 이미지마다 개별 API 호출 수행 (배치가 아닌 순차 처리)
2. try-except 블록으로 각 단계별 에러 캐치
3. JSON 파싱 실패 시 해당 이미지 스킵하고 계속 진행
4. 최종 결과는 성공한 분류만 포함

이는 일부 이미지의 오류가 전체 파이프라인을 중단시키지 않도록 보장합니다."
```

## 결과 분석

### 흥미로운 사례 3개

#### 1. **이미지 #19 - 종합적 자연 장면**
```
분류: Human(Yes), Animal(Yes), Flower(Yes)
근거: "여행객들이 중간 거리에 있고, 사슴과 새가 전체에 퍼져있으며, 
      전경과 주변에 피어난 초목이 있는 포괄적인 자연 장면"
```
**흥미로운 점**: 하나의 이미지에 세 가지 요소가 모두 조화롭게 포함된 경우. 전통 풍경화의 전형적인 구성을 보여줍니다.

#### 2. **이미지 #12 - 신화 속 인물과 생물**
```
분류: Human(Yes), Animal(Yes), Flower(No)
근거: "신화적 장면에서 인물(신이나 영웅)이 신화적 생물과 상호작용하고 있으며,
      중앙 초점은 신성한 존재들이 신화적 짐승을 탈 것"
```
**흥미로운 점**: 모델이 신화적 장면에서 "인물"과 "동물"의 개념을 정확히 이해하고 구분했습니다.

#### 3. **이미지 #15 - 조류 과학 일러스트레이션**
```
분류: Human(No), Animal(Yes), Flower(No)
근거: "다양한 조류 종이 표시되어 가지에 앉아있으며, 
      표본은 프레임의 상단과 중간 부분에 분포"
```
**흥미로운 점**: 과학적 일러스트레이션에서 여러 동물 표본을 정확히 감지하고 공간적으로 설명했습니다.

### 헷갈렸거나 틀렸을 가능성이 있는 사례 2개

#### 1. **이미지 #6 - 사냥 장면의 모호성**
```
분류: Human(Yes), Animal(Yes), Flower(No)
근거: "귀족들이 배경에 말을 타고 있고, 사냥 개들이 전경에 있다.
      인간과 동물이 구성에 필수적이다."
```
**의심점**: 배경에 있는 기마 인물이 명확하게 보이는지의 문제. 그림의 해상도나 스타일에 따라 
배경의 인물을 감지 실패할 수도 있습니다.

#### 2. **이미지 #11 - 식물 분류의 경계**
```
분류: Human(No), Animal(No), Flower(Yes)
근거: "평판 배경에 배치된 이국 식물의 식물학 삽화.
      난초와 기타 열대 꽃이 있는 전체 구성을 채운다."
```
**의심점**: "꽃"의 정의. 난초와 같은 열대 식물이 전통적인 꽃(장미, 백합)과 
구별되는지 확인 필요. 모델이 일반적인 "꽃"보다는 "식물" 전체를 감지했을 가능성.
