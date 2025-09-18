# SKN16-3rd-5Team

**SK 네트웍스 Family AI 캠프 16기 3차 단위프로젝트 · 펫보험 약관 RAG 기반 QA 시스템**

---

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-%230073b3?style=for-the-badge&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-0052CC?style=for-the-badge&logoColor=white)](https://github.com/facebookresearch/faiss)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-9955BB?style=for-the-badge&logoColor=white)](https://docs.trychroma.com/)
[![Gradio](https://img.shields.io/badge/Gradio-171515?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)

---

## 🗂️ 목차
- [팀 소개](#팀-소개)
- [프로젝트 개요](#프로젝트-개요)
- [주제 선정 배경](#주제-선정-배경)
- [폴더 및 파일 구조](#폴더-및-파일-구조)
- [데이터 설명](#데이터-설명)
- [실행 방법](#실행-방법)
- [주요 기능 및 시스템 구성](#주요-기능-및-시스템-구성)
- [프로젝트 산출물](#프로젝트-산출물)
- [프로젝트 결과 및 분석](#프로젝트-결과-및-분석)
- [한계점 및 발전방향](#한계점-및-발전방향)
- [프로젝트 회고](#프로젝트-회고)

---

## 팀 소개

<table>
<thead>
<tr>
<td align="center"><img src="https://i.namu.wiki/i/yBUlarXaiOUlHnIDDEAtvqGIn_gl9auAY0UB6kzsFd3hjLyUAe_le8z_rUI7DLVxJIp7jHThGGtpQJpGCHfkig.webp" width="110"/><br><b>차하경</b><br><span style="font-size:13px">팀장<br>도메인 분석<br>pdf 추출 모델 비교/분석<br>RAG WorkFlow 설계</span></td>
<td align="center"><img src="https://static.wikia.nocookie.net/kimetsu-no-yaiba-fan/images/4/41/Shinobu_anime_design.png/revision/latest?cb=20201006000955" width="110"/><br><b>황하영</b><br><span style="font-size:13px">임베딩 모델 분석<br>모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://i.namu.wiki/i/HbTvNAaTQDJeZgmH8UyOgd9HF2bQ30jgy2gHhmOSqwNphDCS4g3Nw6MO3OTMi84jmwylrle1vpYzJi-xIvu8lg.webp" width="110"/><br><b>진세현</b><br><span style="font-size:13px">도메인 분석<br>초기 모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://i.namu.wiki/i/VcDyzxOl21BA37mCQjUv5B3AeWmSyoWKHbTRfemqLkx3OY67uQdAfX_4F8r11Z21hAcT1ssgTouWQ8Z9vvlXHw.webp" width="110"/><br><b>김나은</b><br><span style="font-size:13px">임베딩 모델 분석<br>모델 테스트<br>최종 보고서 작성</span></td>
<td align="center"><img src="https://avatars.githubusercontent.com/u/190079140?v=4" width="110"/><br><b>문승현</b><br><span style="font-size:13px">프로젝트 기획<br>DB 설계<br>RAG 모델링</span></td>
</tr>
</thead>
</table>

---

## 프로젝트 개요

여러 펫보험사별 약관/상품 데이터를 대상으로 검색·추론(QA)이 결합된 RAG(Retrieval Augmented Generation) 기반 QA 시스템을 구축합니다.  
다양한 모델 및 검색 방법을 실험·비교하여 소비자 접근성 촉진과 산업의 데이터 활용 혁신을 지원합니다.

---

## 주제 선정 배경

|<img src="https://onimg.nate.com/orgImg/ed/2017/11/16/PS17111600045.jpg" width="240">|<img src="https://thumb.mt.co.kr/06/2024/03/2024031114052219364_1.jpg" width="300">|
|:---:|:---:|
|반려동물 시장 성장, 보험 수요 급증<br>시장 성장의 주요 한계(가입률 저조/손해율 高)<br>디지털 기반 접근성 강화 필요|약관 복잡성/정보 비대칭 탓 실소비자 불편<br>자동 QA 시스템을 통한 산업 혁신 시도|

---

## 폴더 및 파일 구조

SKN16-3rd-5Team/
├── data/
│ └── insurance/ # 보험 chunk 데이터 (CSV)
├── peurojegteu3_susususususujeong.ipynb # 전체 분석 및 RAG QA 코드 노트북
├── requirements.txt
├── openaikey.txt
├── assets/
│ ├── 5조_발표자료.pdf # 최종 발표자료
│ └── screens/ # Gradio 데모 UI 스크린샷(선택)

- 주요 코드는 `peurojegteu3_susususususujeong.ipynb`에서 실행

---

## 데이터 설명

- **chunk 기반 보험 약관 데이터(CSV)**: 사별로 chunk/페이지/카테고리 등 메타 정보 포함
- 다양한 보험사 데이터 통합·정제 후 벡터DB/RAG QA에 사용

---

## 실행 방법

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN16-3rd-5Team.git
cd SKN16-3rd-5Team
pip install -r requirements.txt

openaikey.txt에 본인 OpenAI API 키 입력
Jupyter에서 수정바람.ipynb 실행
```

---

## 주요 기능 및 시스템 구성

- **데이터 전처리**: 기관별 chunk 취합/벡터 DB 구축(FAISS, BM25, Chroma 등 지원)
- **RAG QA**: LangChain + OpenAI API + 임베딩/검색 실험 구조
- **UI·데모**: Gradio 실시간 Q&A 챗봇, 피드백 대시보드(선택)
- **정량 평가/시각화**: Hit Rate, MRR, 사용자 피드백 분석

---

## 프로젝트 산출물

- **보험 chunk 데이터**: `data/insurance/`
- **RAG QA 전체 코드·분석**: `peurojegteu3_susususususujeong.ipynb`
- **API Key 관리**: `openaikey.txt`
- **발표자료**: `assets/5조_발표자료.pdf`
- **그레이디오(Gradio) 데모**: `assets/screens/`

---

## 프로젝트 결과 및 분석

- 모델·임베딩·검색별 QA 성능, 실질 답변 정확도 및 사용자 피드백 통계
- 주요 실험 결과/분석 및 시사점은 **발표자료(assets/5조_발표자료.pdf)** 참고

---

## 한계점 및 발전방향

**한계**
- 보험사 자료 불균형·약관 chunk 품질
- 도메인 특성상 텍스트 난해성, 실시간 데이터 반영 한계

**향후 발전**
- chunk 구조 및 QA 고도화, 다양한 DB 지원, API화 등

**기대 효과**
- 자동화된 약관 Q&A, 소비자 정보격차 해소, 산업 내 데이터 기반 혁신 지원

---

## 프로젝트 회고

| 이름   | 한마디 회고/성장포인트                |
| ------ | ----------------------------------- |
| 차하경 |                                     |
| 황하영 |                                     |
| 진세현 |                                     |
| 김나은 |                                     |
| 문승현 |                                     |

---

## 기술스택

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
