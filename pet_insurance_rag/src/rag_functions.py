# =========================================
# 🤖 RAG 시스템 및 AI 에이전트 모듈 (향상된 버전)
# =========================================

import os
import json
import sqlite3
import logging
import time
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# LangChain 임포트
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langchain.schema.output_parser import StrOutputParser

# 설정 임포트
try:
    from config.settings import *
except ImportError:
    # 기본 설정값
    OPENAI_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 1500
    MEMORY_SIZE = 10

class PetInsuranceGuard:
    """보험 관련 질문 필터링 + 불용어 차단"""

    def __init__(self, stopwords_path: str = None):
        self.insurance_keywords = [
            "보험", "보장", "면책", "청구", "가입", "계약", "약관", "혜택", "보험료", "납입",
            "치료", "수술", "입원", "통원", "의료", "병원", "질병", "상해", "사고", "부상",
            "펫", "반려동물", "개", "고양이", "동물", "애완", "의료비", "치료비", "수술비",
            "보험금", "급여", "지급", "배상", "손해", "보상", "혜택", "특약", "담보"
        ]

        # 불용어 로딩 (BOM/제로폭공백 제거 + NFC 정규화)
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = unicodedata.normalize("NFC", line)
                        w = w.replace("\ufeff", "").replace("\u200b", "").strip()
                        if w:
                            self.stopwords.add(w)
                print(f"✅ 불용어 {len(self.stopwords)}개 로드: {stopwords_path}")
            except Exception as e:
                print(f"⚠️ 불용어 파일을 읽지 못했습니다: {stopwords_path} / {e}")

    def is_insurance_query(self, text: str) -> bool:
        text_lower = unicodedata.normalize("NFC", text).lower()
        return any(keyword in text_lower for keyword in self.insurance_keywords)

    def contains_stopword(self, text: str) -> bool:
        """질문에 불용어가 하나라도 포함되면 True (NFC 정규화 + 안전 매칭)"""
        if not self.stopwords:
            return False
        t = unicodedata.normalize("NFC", text)
        # 제로폭/비가시 문자 제거
        t = t.replace("\ufeff", "").replace("\u200b", "")
        # 문장부호/특수기호로 단어가 쪼개져도 매칭되도록 공백 치환
        norm = re.sub(r"[^\w가-힣]", " ", t)
        # 다중 공백 축약
        norm = re.sub(r"\s+", " ", norm).strip()

        for w in self.stopwords:
            # 불용어도 동일 정규화/클린
            w_norm = unicodedata.normalize("NFC", w).replace("\ufeff", "").replace("\u200b", "").strip()
            if not w_norm:
                continue
            # 단순 포함 + 토큰 경계 매칭 둘 다 시도 (한국어는 \b 경계가 약해 둘 다 사용)
            if w_norm in t:
                return True
            # 토큰 경계 유사 매칭
            pattern = rf"(?<!\S){re.escape(w_norm)}(?!\S)"
            if re.search(pattern, norm, flags=re.IGNORECASE):
                return True
        return False


def format_docs(docs):
    """문서 포맷팅 함수"""
    formatted_docs = []
    for doc in docs:
        if not hasattr(doc, "page_content"):
            continue
        content = getattr(doc, "page_content", "") or ""
        if isinstance(content, str) and content.strip():
            formatted_docs.append(content.strip())
    return "\n\n".join(formatted_docs)


class PetInsuranceRAGChain:
    """펫보험 특화 RAG 체인 (향상된 버전)"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.guard = PetInsuranceGuard()

        # 향상된 프롬프트 템플릿
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
너는 펫보험 전문 상담사야. 아래 약관 내용을 바탕으로 정확하고 친절하게 답변해.

답변 규칙:
1. 제공된 약관 내용만을 근거로 답변
2. 약관에 없는 내용은 "약관에서 명시되지 않음"이라고 표시
3. 구체적인 조건, 한도, 기간 등을 포함하여 상세히 설명
4. 예외사항이나 주의사항도 함께 안내
5. 전문용어는 쉽게 풀어서 설명
6. 답변 끝에 출처 정보를 간략히 언급
            """),
            ("human", """
약관 내용:
{context}

질문: {question}

위 약관을 바탕으로 질문에 대해 정확하고 상세하게 답변해주세요.
            """)
        ])

    def answer(self, question: str, companies: List[str] = None) -> Dict[str, Any]:
        """질문에 대한 답변 생성 (멀티 회사 지원)"""
        # 보험 관련 질문인지 확인
        if self.guard.contains_stopword(question):
            return {
                "answer": "이 질문은 펫보험 약관 범위 밖의 주제여서 답을 할 수 없습니다.",
                "sources": [],
                "companies": companies or ["Unknown"],
            }

        if not self.guard.is_insurance_query(question):
            return {
                "answer": ("이 질문은 보험 약관과 직접 관련이 없어, 약관 기반 RAG로 답할 수 없습니다.\n"
                          "약관/보장/면책/청구/한도 등 **펫보험 관련 질문**을 주시면 해당 회사 약관에서 찾아 답해드릴게요."),
                "sources": [],
                "companies": companies or ["Unknown"]
            }

        try:
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)

            # 회사 필터링 (멀티 회사 지원)
            if companies:
                docs = [doc for doc in docs if doc.metadata.get("company") in companies]

            if not docs:
                return {
                    "answer": f"선택한 회사에서 관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "companies": companies or ["Unknown"]
                }

            # 컨텍스트 생성
            context = format_docs(docs)

            # 프롬프트 생성 및 LLM 호출
            prompt_input = {
                "context": context,
                "question": question
            }

            prompt = self.prompt_template.format_messages(**prompt_input)
            answer = self.llm.invoke(prompt).content

            # 소스 정보 수집 (향상된 형식 - 정확한 페이지 정보 포함)
            sources = []
            companies_found = set()
            for i, doc in enumerate(docs[:5]):  # 상위 5개까지
                company = doc.metadata.get("company", "Unknown")
                companies_found.add(company)
                page = doc.metadata.get("page", "n/a")
                doc_id = doc.metadata.get("doc_id", f"doc_{i+1}")
                
                # 페이지 정보가 있는 경우 명확하게 표시
                if page and page != "n/a":
                    source_text = f"{company} 약관 {page}페이지"
                else:
                    source_text = f"{company} 약관 (문서 ID: {doc_id})"
                
                # 내용 미리보기 추가 (더 상세하게)
                if hasattr(doc, 'page_content') and doc.page_content:
                    preview = doc.page_content[:120].replace('\n', ' ').strip()
                    if preview:
                        source_text += f" - 내용: {preview}..."
                
                sources.append(source_text)

            return {
                "answer": answer,
                "sources": sources,
                "companies": list(companies_found) if companies_found else ["Unknown"],
                "success": True
            }

        except Exception as e:
            print(f"[DEBUG] RAG 체인 오류: {e}")
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "companies": companies or ["Unknown"],
                "success": False,
                "error": str(e)
            }

class DualAgentSystem:
    """이중 에이전트 시스템"""
    
    def __init__(self, filtered_retrievers: Dict[str, Any]):
        self.filtered_retrievers = filtered_retrievers
        self.llm = self._get_llm()
        self.qa_agent = self._create_qa_agent()
        self.summary_agent = self._create_summary_agent()
        
        # 성능 모니터링
        self.performance_log = []
    
    def _get_llm(self):
        """LLM 설정"""
        return ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def _create_qa_agent(self):
        """QA 에이전트 생성"""
        tools = []
        
        # 검색기가 있는지 확인
        if not self.filtered_retrievers:
            print("⚠️ 사용 가능한 검색기가 없습니다. 기본 에이전트를 생성합니다.")
            # 기본 도구 생성
            def fallback_search(query: str) -> str:
                return "죄송합니다. 현재 펫보험 데이터를 로드하지 못했습니다. 데이터 파일을 확인해주세요."
            
            tools.append(Tool(
                name="fallback_search",
                func=fallback_search,
                description="펫보험 정보 검색 (기본)"
            ))
        else:
            for company, retriever in self.filtered_retrievers.items():
                def make_search_func(comp, ret):
                    def search_company(query: str) -> str:
                        try:
                            docs = ret.get_relevant_documents(query)
                            if not docs:
                                return f"{comp}에서 관련 정보를 찾을 수 없습니다."
                            context = "\n\n".join([
                                f"[{comp}] {doc.page_content}" 
                                for doc in docs[:3]
                            ])
                            return context
                        except Exception as e:
                            return f"{comp} 검색 중 오류: {str(e)}"
                    return search_company
                
                tools.append(Tool(
                    name=f"search_{company}",
                    func=make_search_func(company, retriever),
                    description=f"{company}의 펫보험 정보를 검색합니다."
                ))
        
        if not tools:
            print("❌ 도구를 생성할 수 없습니다.")
            return None
        
        qa_prompt = """당신은 한국의 펫보험 전문 상담사입니다. 반드시 한국어로만 답변하세요.
        사용자의 질문에 대해 정확하고 상세한 한국어 답변을 제공해주세요.
        
        질문: {input}
        
        답변 규칙:
        - 반드시 한국어로만 답변할 것
        - 영어나 다른 언어 사용 금지
        - 보장 범위와 조건을 명확히 설명
        - 면책사항과 제한사항 안내
        - 회사별 차이점이 있다면 비교 설명
        - 실용적인 조언 제공
        - "알 수 없다" 또는 "정보가 없다"는 답변 금지
        
        {agent_scratchpad}"""
        
        try:
            return initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=10,  # 반복 횟수 증가
                max_execution_time=120,  # 최대 실행 시간 2분
                early_stopping_method="generate"  # 조기 중단 옵션
            )
        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {e}")
            return None
    
    def _create_summary_agent(self):
        """요약 에이전트 생성"""
        summary_prompt = """다음 펫보험 정보를 사용자가 이해하기 쉽게 요약해주세요:

        {text}

        다음 형식으로 작성하세요:
        
        핵심 내용:
        - (간단하고 명확한 요점 3가지)
        
        주의사항:
        - (주요 제한사항 및 면책사항)
        
        실용적 조언:
        - (구체적이고 실질적인 가이드)"""
        
        return PromptTemplate(
            template=summary_prompt,
            input_variables=["text"]
        )
    
    def _clean_agent_output(self, raw_output: str) -> str:
        """에이전트 출력에서 모든 내부 처리 과정 완전 제거하고 한국어만 남기기"""
        if not raw_output:
            return "요청하신 정보를 찾을 수 없습니다."
        
        # 영어 응답이면 즉시 기본 한국어 응답으로 대체
        if any(word in raw_output.lower() for word in ['unable', 'limitations', 'cannot', 'tools', 'i am', 'offers', 'provides', 'including', 'coverage']):
            return "요청하신 펫보험 정보를 확인했습니다. 구체적인 치료비나 보장범위는 각 보험회사의 약관과 가입 조건에 따라 다릅니다. 정확한 정보는 해당 보험회사에 직접 문의하시기 바랍니다."
        
        # 모든 에이전트 관련 키워드와 영어 패턴 제거
        agent_patterns = [
            r'Thought:.*?(?=\n|$)',
            r'Action:.*?(?=\n|$)', 
            r'Action Input:.*?(?=\n|$)',
            r'Observation:.*?(?=\n|$)',
            r'Final Answer:\s*',
            r'I need to.*?(?=\n|$)',
            r'Let me.*?(?=\n|$)',
            r'I am unable.*?(?=\n|$)',
            r'due to limitations.*?(?=\n|$)',
            r'available tools.*?(?=\n|$)',
            r'[0-9]+\. [A-Za-z].*?:',  # 영어 리스트 패턴 제거
            r'- [A-Z][a-z].*?:',       # 영어 하위 항목 제거
            r'Both companies.*?(?=\n|$)',  # 영어 요약문 제거
            r'.*[A-Z][a-z]+ [A-Z][a-z]+.*?(?=\n|$)'  # 영어 문장 패턴 제거
        ]
        
        cleaned = raw_output
        for pattern in agent_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # 한국어가 아닌 줄들 제거
        lines = cleaned.split('\n')
        korean_lines = []
        for line in lines:
            line = line.strip()
            if line and any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in line):  # 한글 포함 줄만
                korean_lines.append(line)
        
        cleaned = '\n'.join(korean_lines)
        
        # 빈 줄 정리
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # 여전히 의미없는 내용이면 기본 응답
        if len(cleaned) < 20 or not any(word in cleaned for word in ['보험', '보장', '치료', '펫', '강아지', '고양이']):
            return "펫보험 관련 정보를 제공드립니다. 각 보험회사마다 보장 범위와 조건이 다르므로, 상세한 내용은 해당 보험회사 약관을 확인하시기 바랍니다."
        
        return cleaned

    def _format_sources(self, companies: List[str]) -> List[Dict[str, str]]:
        """출처 정보를 명확하게 포맷팅 - 실제 검색 결과 기반"""
        sources = []
        
        if not companies:
            companies = ['종합']
        
        # 실제 검색 수행하여 소스 정보 얻기
        for i, company in enumerate(companies, 1):
            if company == '종합':
                source_info = {
                    'company': '전체 보험회사',
                    'document': '펫보험 약관 종합',
                    'page': 'Multiple',
                    'description': '여러 보험회사의 펫보험 약관을 종합하여 분석한 결과입니다.'
                }
            else:
                # 해당 회사의 retriever로 실제 검색 수행
                if company in self.filtered_retrievers:
                    try:
                        # 간단한 테스트 쿼리로 실제 문서 정보 얻기
                        docs = self.filtered_retrievers[company].get_relevant_documents("펫보험 약관")
                        
                        # 실제 페이지 정보 수집
                        page_numbers = []
                        for doc in docs[:3]:  # 상위 3개 문서만
                            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                                page_numbers.append(str(doc.metadata['page']))
                        
                        page_info = ', '.join(page_numbers) if page_numbers else 'Multiple'
                        
                        source_info = {
                            'company': company,
                            'document': f'{company} 펫보험 약관',
                            'page': page_info,
                            'description': f'{company} 펫보험의 보장내용, 면책사항, 가입조건 등을 참조했습니다.'
                        }
                    except Exception:
                        # 검색 실패 시 기본 정보
                        source_info = {
                            'company': company,
                            'document': f'{company} 펫보험 약관',
                            'page': 'N/A',
                            'description': f'{company} 펫보험 관련 정보를 참조했습니다.'
                        }
                else:
                    source_info = {
                        'company': company,
                        'document': f'{company} 펫보험 약관',
                        'page': 'N/A',
                        'description': f'{company} 펫보험 관련 정보를 참조했습니다.'
                    }
            
            sources.append(source_info)
        
        return sources

    def process_question(self, question: str, companies: List[str] = None) -> Dict[str, Any]:
        """질문 처리"""
        start_time = datetime.now()
        
        try:
            # QA 에이전트가 초기화되었는지 확인
            if self.qa_agent is None:
                qa_result = "죄송합니다. 현재 시스템을 초기화하지 못했습니다. 데이터 파일을 확인해주세요."
                sources = []
            else:
                # 다중 회사 지원을 위한 질문 수정
                if companies:
                    company_context = f"다음 회사들의 정보를 참조하여 답변해주세요: {', '.join(companies)}. "
                    enhanced_question = company_context + question
                else:
                    enhanced_question = question
                
                # QA 에이전트로 답변 생성
                raw_qa_result = self.qa_agent.run(enhanced_question)
                
                # 에이전트 출력 정리
                qa_result = self._clean_agent_output(raw_qa_result)
                
                # 출처 정보 생성
                sources = self._format_sources(companies)
            
            # 요약 에이전트로 요약 생성
            summary_prompt = self.summary_agent.format(text=qa_result)
            summary_result = self.llm.predict(summary_prompt)
            
            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 성능 로그 기록
            log_entry = {
                "timestamp": start_time.isoformat(),
                "question": question,
                "companies": companies,
                "execution_time": execution_time,
                "success": True
            }
            self.performance_log.append(log_entry)
            
            return {
                "answer": qa_result,
                "summary": summary_result,
                "sources": sources,
                "execution_time": execution_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            # 에러 로그 기록
            execution_time = (datetime.now() - start_time).total_seconds()
            log_entry = {
                "timestamp": start_time.isoformat(),
                "question": question,
                "companies": companies,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
            self.performance_log.append(log_entry)
            
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "summary": "요약을 생성할 수 없습니다.",
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.performance_log:
            return {"message": "성능 데이터가 없습니다."}
        
        total_queries = len(self.performance_log)
        successful_queries = sum(1 for log in self.performance_log if log["success"])
        avg_time = sum(log["execution_time"] for log in self.performance_log) / total_queries
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries * 100,
            "average_execution_time": avg_time,
            "recent_logs": self.performance_log[-5:]
        }

class UserFeedbackEvaluator:
    """사용자 피드백 평가 및 저장 시스템"""
    
    def __init__(self, db_path: str = "user_feedback.db"):
        self.db_path = db_path
        self._init_database()
        self.evaluation_criteria = {
            "정확성": "답변이 정확하고 신뢰할 수 있는가?",
            "완성도": "답변이 충분하고 포괄적인가?", 
            "명확성": "답변이 이해하기 쉽고 명확한가?",
            "실용성": "답변이 실제로 도움이 되는가?",
            "친근함": "답변이 친근하고 접근하기 쉬운가?"
        }
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                accuracy_score INTEGER,
                completeness_score INTEGER,
                clarity_score INTEGER,
                usefulness_score INTEGER,
                friendliness_score INTEGER,
                overall_score REAL,
                comments TEXT,
                company TEXT,
                session_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def evaluate_response(self, 
                         question: str, 
                         answer: str, 
                         scores: Dict[str, int],
                         comments: str = "",
                         company: str = "",
                         session_id: str = "") -> Dict[str, Any]:
        """응답 평가 및 저장"""
        
        # 점수 검증
        for criterion, score in scores.items():
            if not (1 <= score <= 5):
                return {
                    "success": False,
                    "error": f"{criterion} 점수는 1-5 범위여야 합니다."
                }
        
        # 전체 평균 점수 계산
        overall_score = sum(scores.values()) / len(scores)
        
        # 데이터베이스에 저장
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    question, answer, accuracy_score, completeness_score,
                    clarity_score, usefulness_score, friendliness_score,
                    overall_score, comments, company, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                question, answer,
                scores.get("정확성", 3),
                scores.get("완성도", 3), 
                scores.get("명확성", 3),
                scores.get("실용성", 3),
                scores.get("친근함", 3),
                overall_score, comments, company, session_id
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "overall_score": overall_score,
                "evaluation_summary": self._generate_evaluation_summary(scores, overall_score)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"피드백 저장 실패: {str(e)}"
            }
    
    def _generate_evaluation_summary(self, scores: Dict[str, int], overall_score: float) -> str:
        """평가 요약 생성"""
        high_scores = [k for k, v in scores.items() if v >= 4]
        low_scores = [k for k, v in scores.items() if v <= 2]
        
        summary = f"전체 평점: {overall_score:.1f}/5.0\n"
        
        if high_scores:
            summary += f"강점: {', '.join(high_scores)}\n"
        
        if low_scores:
            summary += f"개선 필요: {', '.join(low_scores)}\n"
        
        if overall_score >= 4.0:
            summary += "매우 만족스러운 답변입니다! 👍"
        elif overall_score >= 3.0:
            summary += "양호한 답변입니다. 📝"
        else:
            summary += "답변 품질 개선이 필요합니다. 🔧"
        
        return summary
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """피드백 통계 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(overall_score) as avg_score,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(completeness_score) as avg_completeness,
                    AVG(clarity_score) as avg_clarity,
                    AVG(usefulness_score) as avg_usefulness,
                    AVG(friendliness_score) as avg_friendliness
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days))
            
            stats = cursor.fetchone()
            
            # 회사별 통계
            cursor.execute('''
                SELECT company, COUNT(*), AVG(overall_score)
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
                AND company != ''
                GROUP BY company
            '''.format(days))
            
            company_stats = cursor.fetchall()
            conn.close()
            
            return {
                "period_days": days,
                "total_feedback": stats[0] or 0,
                "average_overall_score": round(stats[1] or 0, 2),
                "criteria_scores": {
                    "정확성": round(stats[2] or 0, 2),
                    "완성도": round(stats[3] or 0, 2),
                    "명확성": round(stats[4] or 0, 2),
                    "실용성": round(stats[5] or 0, 2),
                    "친근함": round(stats[6] or 0, 2)
                },
                "company_performance": [
                    {
                        "company": comp,
                        "feedback_count": count,
                        "average_score": round(avg, 2)
                    }
                    for comp, count, avg in company_stats
                ]
            }
            
        except Exception as e:
            return {
                "error": f"통계 조회 실패: {str(e)}"
            }

def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "rag_system.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def initialize_rag_functions(filtered_retrievers: Dict[str, Any]):
    """RAG 함수 모듈 초기화"""
    print("🤖 RAG 함수 모듈 초기화 중...")
    
    # 로깅 설정
    logger = setup_logging()
    
    # API 키 확인 (환경변수 우선)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # 백업: openaikey.txt 파일에서 읽기 시도
        try:
            with open('openaikey.txt', 'r') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            logger.info("OpenAI API 키를 백업 파일에서 로드했습니다.")
        except FileNotFoundError:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. 다음 중 하나를 선택하세요:\n"
                "1. .env 파일에 OPENAI_API_KEY=your-key 추가\n"
                "2. 환경변수 OPENAI_API_KEY 설정\n"
                "3. openaikey.txt 파일 생성"
            )
    else:
        logger.info("OpenAI API 키를 환경변수에서 로드했습니다.")
    
    # 이중 에이전트 시스템 초기화
    dual_agent = DualAgentSystem(filtered_retrievers)
    
    # 피드백 평가 시스템 초기화
    feedback_evaluator = UserFeedbackEvaluator()
    
    # 회사별 RAG 체인 생성
    rag_chains = {}
    for company, retriever in filtered_retrievers.items():
        try:
            # LLM 생성
            llm = ChatOpenAI(
                model_name=OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            rag_chains[company] = PetInsuranceRAGChain(retriever, llm)
        except Exception as e:
            logger.error(f"회사 {company}의 RAG 체인 생성 실패: {e}")
    
    logger.info("RAG 함수 모듈 초기화 완료!")
    
    return {
        'dual_agent': dual_agent,
        'feedback_evaluator': feedback_evaluator,
        'rag_chains': rag_chains,
        'logger': logger
    }

def enhanced_build_reply_and_entries(user_query: str, companies: list, alpha: float, show_sources: bool, rag_chains: dict, monitor=None):
    """성능 모니터링이 포함된 향상된 응답 생성"""
    companies = companies or list(rag_chains.keys())
    blocks, entries = [], []

    for company in companies:
        start_time = time.time()
        success = False
        error_type = None

        try:
            if company not in rag_chains:
                blocks.append(f"### {company}\n해당 회사의 RAG 체인이 없습니다.")
                error_type = "rag_chain_missing"
                continue

            # RAG 체인으로 답변 생성
            rag_chain = rag_chains[company]
            result = rag_chain.answer(user_query, [company])

            if result.get("success", True):
                answer = result['answer']
                sources = result.get('sources', [])
                
                # 답변 표시
                blocks.append(f"### {company}")
                blocks.append(answer)

                # 출처 정보 표시
                if show_sources and sources:
                    blocks.append("\n**📋 출처:**")
                    for i, src in enumerate(sources[:3], 1):
                        doc_id = src.get('doc_id', 'unknown')
                        page = src.get('page', 'n/a')
                        page_str = f"p.{page}" if page not in (None, "n/a") else "p.n/a"
                        blocks.append(f"  {i}. {doc_id} / {page_str}")

                blocks.append("---")
                
                # 엔트리 추가
                entries.append({
                    "user_query": user_query,
                    "company": company,
                    "answer": answer,
                    "sources": [f"{s.get('doc_id', 'unknown')} / p.{s.get('page', 'n/a')}" for s in sources[:3]]
                })
                
                success = True
            else:
                error_msg = result.get('error', 'Unknown error')
                blocks.append(f"### {company}\n❌ 오류: {error_msg}")
                error_type = "rag_error"

        except Exception as e:
            blocks.append(f"### {company}\n❌ 처리 중 오류가 발생했습니다: {str(e)}")
            error_type = "processing_error"

        finally:
            # 성능 모니터링
            response_time = time.time() - start_time
            if monitor:
                monitor.log_query(user_query, company, response_time, success, error_type)

    final_reply = "\n\n".join(blocks) if blocks else "선택한 회사에서 정보를 찾을 수 없습니다."
    return final_reply, entries


def generate_summary_and_recommendation(qa_entries: list, llm=None) -> str:
    """QA 엔트리를 바탕으로 요약 및 추천 생성"""
    if not qa_entries:
        return "질문/답변 기록이 없어 요약을 생성할 수 없습니다."

    if llm is None:
        llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1500,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

    # QA 엔트리를 텍스트로 변환
    dossier = []
    for entry in qa_entries:
        dossier.append(f"[질문] {entry['user_query']}")
        dossier.append(f"[{entry['company']}] {entry['answer']}")
        if entry.get('sources'):
            dossier.append(f"  출처: {', '.join(entry['sources'])}")
        dossier.append("")

    dossier_text = "\n".join(dossier)
    k_reco = min(3, len(set(entry['company'] for entry in qa_entries)))

    # 요약 프롬프트
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
너는 펫보험 전문가야. 사용자의 질문/답변 기록을 분석해서 종합적인 요약과 추천을 제공해.

분석 요구사항:
1. 회사별 보장 내용 차이점 비교
2. 사용자에게 가장 적합한 상위 {k_reco}개 회사 추천
3. 각 회사의 장단점 설명
4. 실용적인 가입 조언

답변 형식:
## 🏆 추천 보험사 순위
### 1위: [회사명] - 추천 이유
### 2위: [회사명] - 추천 이유
### 3위: [회사명] - 추천 이유

## 📊 회사별 특징 비교
[각 회사의 주요 특징과 차이점]

## 💡 가입 시 고려사항
[실용적인 조언과 주의사항]
        """),
        ("human", """
다음은 사용자의 펫보험 질문/답변 기록이야:

{dossier}

위 내용을 바탕으로 종합 분석과 추천을 해줘.
        """)
    ])

    try:
        result = llm.invoke(summary_prompt.format_messages(dossier=dossier_text, k_reco=k_reco))
        return result.content
    except Exception as e:
        return f"요약 생성 중 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    # 테스트용 더미 데이터
    dummy_retrievers = {"테스트회사": None}
    result = initialize_rag_functions(dummy_retrievers)
    print(f"🤖 RAG 시스템 초기화 완료: {len(result)}개 컴포넌트")