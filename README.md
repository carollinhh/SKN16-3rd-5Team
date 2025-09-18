# SKN16-3rd-5Team
SKN 16기 3차 단위프로젝트

### 팀 소개

|<img src="https://i.namu.wiki/i/yBUlarXaiOUlHnIDDEAtvqGIn_gl9auAY0UB6kzsFd3hjLyUAe_le8z_rUI7DLVxJIp7jHThGGtpQJpGCHfkig.webp" width="155" height="165"> | <img src="https://static.wikia.nocookie.net/kimetsu-no-yaiba-fan/images/4/41/Shinobu_anime_design.png/revision/latest?cb=20201006000955" width="155" height="200"> | <img src="https://i.namu.wiki/i/HbTvNAaTQDJeZgmH8UyOgd9HF2bQ30jgy2gHhmOSqwNphDCS4g3Nw6MO3OTMi84jmwylrle1vpYzJi-xIvu8lg.webp" width="155" height="170"> | <img src="https://i.namu.wiki/i/aJ8BIe4CcPyG7D1qKxbLzIOEwcNKP5RsCmb_POFJ-MbAInDE8dK0XvYVA-3ZvADJKJpey8LtqlhJNTOvrrGq8g.webp" width="155" height="170"> | <img src="https://i.namu.wiki/i/VcDyzxOl21BA37mCQjUv5B3AeWmSyoWKHbTRfemqLkx3OY67uQdAfX_4F8r11Z21hAcT1ssgTouWQ8Z9vvlXHw.webp" width="170" height="170"> |
|---|---|---|---|---|
|차하경|황하영|진세현|문승현|김나은|
|팀장, 도메인 분석, pdf 텍스트 추출 모델 비교/분석, RAG WorkFlow 설계|임베딩 모델 비교/분석, 모델 테스트, 최종 보고서 작성|도메인 분석, 초기 모델 테스트, 최종 보고서 작성|프로젝트 기획, DB 설계, RAG 모델링|임베딩 모델 비교/분석, 모델 테스트, 최종 보고서 작성|

<br>

## 프로젝트 개요
이 프로젝트는 여러 펫보험사별 약관/상품 문서 데이터를 효과적으로 활용해,  
검색·추론(QA)이 결합된 RAG(Retrieval-Augmented Generation) 시스템을 구축하고 성능을 비교·분석함으로써,
소비자 접근성 제고 방안을 모색하고 펫보험 산업의 지속가능한 성장 전략을 제시하는 데 목적이 있습니다.


### 주제 선정 배경
|<img src="https://onimg.nate.com/orgImg/ed/2017/11/16/PS17111600045.jpg" width="300" height="500">|<img src="https://thumb.mt.co.kr/06/2024/03/2024031114052219364_1.jpg" width="400" height="500">
|:---:|:---:|
반려동물 양육 인구 증가로 보험 수요가 확대되고 있으나, 낮은 가입률과 높은 손해율이 시장 성장을 제약하고 있습니다. <br>
이에 따라 디지털 플랫폼 기반 소비자 접근성 강화를 통해 시장 성장을 촉진하고자 했습니다.
펫보험 약관 RAG 기반 챗봇 개발은 그 첫 번째 발걸음이 될 것입니다.


<br>
## 폴더/파일 구조
SKN16-3rd-5Team/
├── 📂 data/
│   ├── 📂 insurance/                    # 보험 데이터
└── 노트북 코드       # RAG 시스템
```

<br>

### 노트북 코드 (상세 분석 과정)
- **기능**: `적어주세요`

## 데이터 설명

- 각 CSV 파일은 보험사별 약관/보장내용을 청크(chunk) 단위로 정제한 데이터입니다.
- 주요 컬럼 예시: chunk 텍스트, 문서/페이지/항목, 등
- 다양한 보험사 데이터를 통합적으로 학습, 검색, QA 기반에 활용합니다.

<br>

## 실행 방법

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN16-3st-5Team.git

cd SKN16-3st-5Team

pip install -r requirements.txt

python .ipynb
```
## 프로젝트 산출물

- **인공지능 데이터 전처리**: `data/insurance` 폴더 내 청킹된 보험 약관 텍스트 데이터
- **펫보험 약관 Q&A 챗봇**: `.ipynb` - RAG 기반 챗봇(주피터 노트북)
- **발표자료**: 프로젝트 전 과정 포함한 종합 발표 자료 `assets/5조_발표자료.pdf`

### 옵션 산출물
- **분석 화면 (Gradio)**: 펫보험 약관 Q&A 챗봇 (화면구성도 `assets/screens` 참고 - 추가 필요)

<br>

## 주요 기능 및 구성

- **데이터 전처리**: 보험사별 chunk 취합, Vector DB 구축
- **임베딩/검색**: FAISS, BM25, Chroma 등 다양한 백엔드 지원
- **QA 시스템**: LangChain 기반 RAG, OpenAI GPT API 활용
- **자동 평가/대시보드**: Hit Rate, MRR 등 정량평가, 피드백 시각화
- **실사용 UI**: Gradio 기반 챗봇, 실시간 평가 및 피드백 수집

<br>

## 결과 및 분석

(발표자료 기반으로 내용 추가)

<br>


### 현재 한계점
1. **추가 필요**: 


### 향후 발전 계획
1. **추가 필요**: 


### 기대 효과
- **추가 필요**: 



## 프로젝트 회고

### 개인별 소감 및 성장 포인트
- **차하경 👑**:


- **황하영**:

  
- **진세현**:

  
- **문승현**:

  
- **김나은**:

  
