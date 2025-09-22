# 펫보험 들5조 - 펫 보험 Q&A 챗봇 서비스

**SK 네트웍스 Family AI 캠프 16기 3차 단위프로젝트**  
📌 펫보험 약관 RAG 기반 QA 시스템

---
## 👥 팀 소개

<table>
<thead>
<tr>
<td align="center"><img src="https://i.namu.wiki/i/yBUlarXaiOUlHnIDDEAtvqGIn_gl9auAY0UB6kzsFd3hjLyUAe_le8z_rUI7DLVxJIp7jHThGGtpQJpGCHfkig.webp" width="110"/><br><b>차하경</b><br><span style="font-size:13px">팀장<br>도메인 분석<br>pdf 추출 모델 비교/분석<br>RAG WorkFlow 설계 및 모델링</span></td>
<td align="center"><img src="https://static.wikia.nocookie.net/kimetsu-no-yaiba-fan/images/4/41/Shinobu_anime_design.png/revision/latest?cb=20201006000955" width="110"/><br><b>황하영</b><br><span style="font-size:13px">임베딩 모델 분석<br>모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://i.namu.wiki/i/HbTvNAaTQDJeZgmH8UyOgd9HF2bQ30jgy2gHhmOSqwNphDCS4g3Nw6MO3OTMi84jmwylrle1vpYzJi-xIvu8lg.webp" width="110"/><br><b>진세현</b><br><span style="font-size:13px">도메인 분석<br>초기 모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://i.namu.wiki/i/VcDyzxOl21BA37mCQjUv5B3AeWmSyoWKHbTRfemqLkx3OY67uQdAfX_4F8r11Z21hAcT1ssgTouWQ8Z9vvlXHw.webp" width="110"/><br><b>김나은</b><br><span style="font-size:13px">임베딩 모델 분석<br>모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://avatars.githubusercontent.com/u/190079140?v=4" width="110"/><br><b>문승현</b><br><span style="font-size:13px">프로젝트 기획<br>DB 설계<br>RAG 모델링</span></td>
</tr>
</thead>
</table>

---
## 📘 프로젝트 개요

본 프로젝트는 펫보험 약관 데이터에 대해 **LangChain + OpenAI API** 기반 RAG QA 시스템을 구현한 사례입니다. 

다양한 임베딩 모델(Sentence-BERT, KoSBERT 등)과 검색 기법(FAISS, Chroma, BM25)을 적용하여, 보험 약관 질의에 대한 정확도 및 응답 품질을 정량 평가(Hit Rate, MRR)와 사용자 피드백을 통해 검증했습니다.  

Gradio 기반 UI 데모를 통해 실시간 Q&A 서비스를 제공하며, 산업 데이터 활용 가능성을 실험했습니다.

---
## 🎯 주제 선정 배경

|<img src="https://onimg.nate.com/orgImg/ed/2017/11/16/PS17111600045.jpg" width="240">|<img src="https://thumb.mt.co.kr/06/2024/03/2024031114052219364_1.jpg" width="300">|
|:---:|:---:|
|반려동물 시장 성장, 보험 수요 급증<br>시장 성장의 주요 한계(가입률 저조/손해율 高)<br>디지털 기반 접근성 강화 필요|약관 복잡성/정보 비대칭 탓 실소비자 불편<br>자동 QA 시스템을 통한 산업 혁신 시도|

"반려동물 시장의 지속가능한 성장을 위해서는 자기부담률 설정 등을 통한 손해율 관리, 규제의 비례성 적용을 통한 소액단기전문보험사의 시장 진입 촉진, **디지털 플렛폼을 활용한 소비자 접근성 제고 노력이 필요**"

---
## ⭐ 주요 기능

| 기능 | 설명 |
|------|------|
| 🤖 **이중 AI 에이전트** | QA 에이전트 + 요약 에이전트로 정확한 답변 |
| 🏢 **회사별 검색** | 특정 보험회사 또는 전체 검색 지원 |
| 📊 **실시간 평가** | 사용자 피드백 + GPT 품질 평가 |
| 🔍 **출처 표시** | 참조 약관과 페이지 번호 명시 |
| 📈 **성능 모니터링** | A/B 테스트 및 실시간 지표 추적 |


---
## 📁 프로젝트 구조

```
pet_insurance_rag/
├── 🚀 main.py                 # 실행 파일
├── 📦 src/                    # 소스 코드
│   ├── data_processing.py     # 데이터 처리 & 벡터 DB
│   ├── rag_functions.py       # RAG 핵심 로직 & AI 에이전트
│   ├── gradio_interface.py    # 웹 인터페이스
│   └── evaluation.py          # 평가 시스템
├── 📄 data/                   # 보험회사 CSV 데이터 (8개)
│   ├──DB손해보험_다이렉트_펫블리_반려견보험.csv
│   ├── KBdirect_금쪽같은_펫보험.csv
│   ├── meritz_펫퍼민트.csv
│   ├── 삼성화재_반려견보험_애니펫.csv
│   ├── 삼성화재_위풍댕댕.csv
│   ├── 삼성화재_착한펫보험.csv
│   ├── 하나펫사랑보험.csv
│   └── 현대해상_굿앤굿우리펫보험.csv
├── ⚙️ config/                 # 설정 파일
│   ├── requirements.txt       # 패키지 목록
│   └── settings.py            # 시스템 설정
├── 🧪 tests/                  # 테스트 파일
├── 📝 logs/                   # 시스템 로그
├── 💾 embeddings_cache/       # 벡터 임베딩 캐시
├── 🗄️ user_feedback.db        # 사용자 피드백 DB
└── 📋 .env.template           # 환경 변수 템플릿
```
---
## 🧭 Work Flow
<img width="817" height="840" alt="workflow" src="https://github.com/user-attachments/assets/668478b3-8b4a-4a62-b048-330aab6b663b" />

## 📊 PDF 추출 방식 평가 지표
<img width="1297" height="237" alt="pdf추출방식" src="https://github.com/user-attachments/assets/7206cb98-4ad0-4b34-9f8b-7acfb8a3b7fb" />

| 커버리지(한글+숫자)|원본의 한글, 숫자가 얼마나 살아 남았는지(빠짐 여부)|
| ------ | ----------------------------------- |
| **CER 전체** | **전체 문자 기준의 오탈, 변형률**|
|**CER 한글+숫자**|**한글, 숫자만 놓고 본 오류물(핵심 정확도)**|
|**숫자보존율**|**금액, 비율, 기간 등 숫자 토큰이 얼마나 보존됐는지**|
|**조항마커 보존율**|**제1조, (1), 1), ① 같은 구조 표식 보존**|
|**불릿 보존율**|**• · – — ● ○ ▪︎ ▶ 등 리스트 불릿 보존**|
|**구두점 JSD**|**구두점 분포의 차이(문장부호 패턴)**|
|**종합점수**|**위 지표들을 가중합한 0–100점**|

---

## 🧱 청킹 
**입력: 보험 약관 원문**
- 단락 구분(조-항-절 체계 분석), 슬라이딩 윈도우로 의미 단위 분할
- 각 청크마다 보장/면책/주체/절차 등 보험 실무 특화 키워드 자동 태깅
- 면책·지급 등 핵심 문장, 참조조항, 페이지, 프리뷰추출 등 메타 자동 분석

파라미터: 최소 250자 ~ 최대 1200자 & 슬라이딩 윈도우 오버랩 100자
**구간정보, 핵심 문장, 근거, 주제, 절차 등 메타 포함 CSV 생성**

---

## 🔎 임베딩 모델 비교/선정
**Query: "사고 치료가 보험기간 말에 걸치면 만료 후 치료도 보상되나요?"**
<img width="782" height="247" alt="임베딩 모델비교" src="https://github.com/user-attachments/assets/dc6e774a-bce2-46c2-94d0-724983835868" />

**OPEN AI 임베딩 모델 선택**

---
## 🔁 리랭크 모델 비교/선정
|모델 이름| 강점 |적합한 케이스|
| ------ | ------------------|----------------- |
|BAAI/bge-reranker-v2-m3| 검색/RAG 정확도 최상급|RAG 파이프라인, 보험·법률 FAQ, 정확도가 중요한 서비스|
|mixedbread-ai/mxbai-rerank-base-v1|범용성, 다양한 태스크|FAQ 매칭, 추천, 다국어 서비스, 빠른 실험|
|jinaai/jina-reranker-v2-base-multilingual|속도-성능 밸런스|실시간 검색/추천, 프로덕션 서비스|

**BAAI/bge-reranker-v2-m3 Rerank 모델 선정**

---
## ⚙️ 실행 방법

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN16-3rd-5Team.git
cd SKN16-3rd-5Team
pip install -r requirements.txt

openaikey.txt에 본인 OpenAI API 키 입력
Jupyter에서 수정바람.ipynb 실행
```
---
## 🧠 LangGraph
<img width="831" height="1421" alt="langgraph3" src="https://github.com/user-attachments/assets/5122b707-a419-4115-8d02-40d07d20cac4" />


---

## 🔍 주요 기능 및 시스템 구성

- **데이터 전처리**: 기관별 chunk 취합/벡터 DB 구축(FAISS, BM25, Chroma 등 지원)
- **RAG QA**: LangChain + OpenAI API + 임베딩/검색 실험 구조
- **UI·데모**: Gradio 실시간 Q&A 챗봇, 피드백 대시보드(선택)
- **정량 평가/시각화**: Hit Rate, MRR, 사용자 피드백 분석

---
## 📦 프로젝트 산출물

- **보험 chunk 데이터**: `data/`
- **RAG QA 전체 코드·분석**: `src/rag_functions.py/`
- **API Key 관리**: `openaikey.txt`
- **발표자료**: `assets/5조_발표자료.pdf`
- **그레이디오(Gradio) 데모**: `src/gradio_interface.py/`
  
---
## 💬 질문/답변
### 1. 보험사 선택
<img width="497" height="420" alt="질문1" src="https://github.com/user-attachments/assets/c779f2e4-6d31-4fe5-b4be-901fa434947b" />

### 2. 답변
<img width="533" height="283" alt="질문2" src="https://github.com/user-attachments/assets/08ac8c7c-4aac-45ff-aea0-65c69740f390" />

### 3. 응답 분석
<img width="448" height="127" alt="질문3" src="https://github.com/user-attachments/assets/cd3ad9e9-220c-4862-8d36-b281ea814551" />

### 펫보험을 제외한 질문 예시
<img width="952" height="127" alt="질문4" src="https://github.com/user-attachments/assets/785bc44c-28cc-4025-ac24-c901e0b7f840" />

Q: 아이스크림 있어?

A: 이 질문은 보험 약관과 직접 관련이 없어, 약관 기반 RAG로 답할 수 없습니다. 약관/보장/면책/청구/한도 등 펫보험 관련 질문을 주시면 해당 회사 약관에서 찾아 답해드릴게요

### 무관한 질문에 대한 가드 적용
<img width="947" height="147" alt="질문5" src="https://github.com/user-attachments/assets/e41a01b9-606d-44ed-bfd5-0e6c795dcaf3" />

Q: 자동차 보험에 대해 보상은 어떻게 받을 수 있어?

A: 이 질문은 펫보험 약관 범위 밖의 주제여서 답변하지 않습니다. 펫보험 약관에 해당 관련 정보가 없습니다

---
## 🎯 프로젝트 결과 및 분석

- 모델·임베딩·검색별 QA 성능, 실질 답변 정확도 및 사용자 피드백 통계
- 주요 실험 결과/분석 및 시사점은 **발표자료(assets/5조_발표자료.pdf)** 참고

---
## 🧭 시사점
- Retriever & GPT API로 RAG 구축 -> 사용자 필요에 **알맞은 응답 제공**
- LangChain 활용 -> **캡슐, 모듈화** 통해 WorkFlow 최적화와 보안 강점
- 사용자 피드백 & 시스템 평가 기능을 통해 사용자, GPT의 **수치적, 정성적 피드백 자료 제공** -> 서비스 이용자 뿐만 아니라 서비스 제공자인 보험 도메인 종사자에게도 **실효성 기대**

---
## 💬 요약/추천
<img width="1047" height="460" alt="요약" src="https://github.com/user-attachments/assets/795ec2bd-9c00-4b33-afa5-fd3d477d7742" />

---
## 📝 답변

<img width="575" height="293" alt="답변1" src="https://github.com/user-attachments/assets/39c0d2ce-a562-47c0-8a64-5d599e4ad5ab" />

<img width="438" height="355" alt="답변2" src="https://github.com/user-attachments/assets/a47bd3d8-3edf-465b-8b8b-b9850e9a739f" />

<img width="292" height="267" alt="답변3" src="https://github.com/user-attachments/assets/76cfd49f-8edb-40ac-9a5d-0cadc0f709e5" />

### sqlite3 활용하여 저장

<img width="1426" height="250" alt="답변4" src="https://github.com/user-attachments/assets/f38ec4d3-4512-49f8-949b-ad86e96af3b1" />

---
## 📈 성능 대시보드

<img width="612" height="481" alt="성능대시보드1" src="https://github.com/user-attachments/assets/aceeca97-9656-4262-9aa9-50a21df49f41" />

<img width="327" height="438" alt="성능대시보드2" src="https://github.com/user-attachments/assets/0510cb11-ab84-4450-bcf5-fe75bb538727" />

총 요청 쿼리 개수, 성공/에러율, 평균 응답 속도 등 포함 -> **실시간 동작 상태 확인 가능**

---

## 🔬 품질 평가

<img width="935" height="437" alt="품질평가2" src="https://github.com/user-attachments/assets/633e4b45-6be3-447d-a4fb-2a844ad8569b" />

<img width="532" height="480" alt="품질평가" src="https://github.com/user-attachments/assets/42b2eed2-aae5-46c6-a0db-adf2559a1e20" />

저장된 테스트 쿼리 랜덤 질의응답(회사 별) -> **GPT 평가**

---

## 🖥️ AB 테스트

<img width="1552" height="385" alt="ab테스트2" src="https://github.com/user-attachments/assets/d0504b68-510b-42d6-967c-0060e5eb6551" />

<img width="1132" height="488" alt="ab테스트1" src="https://github.com/user-attachments/assets/afb0afe1-f031-447d-a287-f2a490874bdc" />

서로 다른 설정의 RAG 시스템 성능 비교 -> **AB테스트**

---
## 🧩 한계점 및 발전방향

**한계**
- 보험사 자료 불균형·약관 chunk 품질
- 도메인 특성상 텍스트 난해성, 실시간 데이터 반영 한계
- 문서, 약관 품질 자동 업데이트 기능 없음 -> 수시로 **문서 최신화 필요**
- 보험료 등 실거래 데이터는 약관으로 확정이 불가 -> **실시간 정보 부재**
- 결국 최종 가입을 위해서는 전문 상담과 약관 원문이 필요 -> **단순 흥미 유발**에 그칠 가능

**향후 발전**
- chunk 구조 및 QA 고도화, 다양한 DB 지원, API화 등
- 보험사 사이트 크롤링을 통해 문서, 약관 품질 자동 업데이트 -> **자동 약관 최신화**
- 만족도, 코멘트 기반 감성/토픽 분석 및 저장 -> **검색 파라미터 자동 튜닝**
- 구조화, 자동화 기능 탑재 -> 실제 가입 의사결정까지 자연스럽게 이어질 수 있도록 **구조 재설계**

**기대 효과**
- 자동화된 약관 Q&A, 소비자 정보격차 해소, 산업 내 데이터 기반 혁신 지원

---

## 📝 프로젝트 회고
| 이름   | 한마디 회고/성장포인트                |
| ------ | ----------------------------------- |
| 차하경 |이번 프로젝트에서는 복잡한 보험 약관 데이터를 RAG 시스템에 적용하며, 텍스트 전처리와 임베딩·검색 기법의 차이가 결과에 큰 영향을 미친다는 점을 체감했습니다. 초반에는 데이터의 불균형과 난해한 용어 때문에 어려움이 있었지만, 팀원들과 함께 다양한 모델과 벡터 DB를 실험하며 문제를 해결할 수 있었습니다. 그 과정에서 단순히 기능 구현을 넘어서, 실제 도메인 데이터를 활용할 때 고려해야 할 요소와 RAG의 장단점을 깊이 이해하게 되었습니다. 무엇보다 협업 속에서 각자의 강점을 발휘하며 성장할 수 있었던 경험이 가장 큰 성과였습니다.|
| 황하영 |RAG 기반 챗봇 프로젝트를 진행하며 지식으로만 알던 내용을 실제로 적용할 수 있어서 좋았습니다. 추후에는 파인튜닝까지도 진행해 보고 싶습니다.|
| 진세현 |                                     |
| 김나은 |                                     |
| 문승현 |특정 도메인(보험)을 대상으로 주제를 정하고 프로젝트를 진행하면서 RAG의 전반적인 동작 방식을 이해하고, 의사소통 및 문제해결능력을 함양했다.|

---
## 🛠️ 기술스택

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0073b3?style=flat-square&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-0052CC?style=flat-square&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-9955BB?style=flat-square&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-171515?style=flat-square&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/tqdm-FFC107?style=flat-square&logoColor=white"/>
</p>

---
