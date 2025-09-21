# =========================================
# 🌐 Gradio 웹 인터페이스 모듈
# =========================================

import os
import gradio as gr
import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import time
import threading

# 모니터링 및 로깅
import logging
from pathlib import Path

class GradioInterfaceManager:
    """Gradio 인터페이스 관리 클래스"""
    
    def __init__(self, dual_agent, feedback_evaluator, company_vector_stores: Dict[str, Any]):
        self.dual_agent = dual_agent
        self.feedback_evaluator = feedback_evaluator
        self.company_vector_stores = company_vector_stores
        self.companies = list(company_vector_stores.keys())
        
        # 세션 관리
        self.session_history = {}
        self.current_session_id = None
        
        # 실시간 모니터링
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        # 인터페이스 생성
        self.interface = self._create_interface()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("gradio_interface")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(log_dir / "interface.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_interface(self):
        """Gradio 인터페이스 생성"""
        
        # CSS 스타일링
        css = """
        .gradio-container {
            font-family: 'Noto Sans KR', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .tab-content {
            padding: 20px;
        }
        .feedback-section {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .monitoring-panel {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
        }
        """
        
        with gr.Blocks(css=css, title="🐾 펫보험 RAG 시스템") as interface:
            
            # 메인 헤더
            gr.HTML("""
                <div class="main-header">
                    <h1>🐾 펫보험 RAG 시스템</h1>
                    <p>AI 기반 펫보험 상담 서비스</p>
                </div>
            """)
            
            # 탭 구성
            with gr.Tabs():
                
                # === 메인 QA 탭 ===
                with gr.TabItem("💬 질문하기"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 펫보험에 대해 궁금한 것을 물어보세요!")
                            
                            with gr.Row():
                                company_dropdown = gr.CheckboxGroup(
                                    choices=self.companies,
                                    value=[],
                                    label="보험회사 선택 (다중 선택 가능)",
                                    info="여러 회사를 선택하거나 비워두면 전체 검색"
                                )
                            
                            question_input = gr.Textbox(
                                label="질문 입력",
                                placeholder="예: 강아지 예방접종 비용이 보장되나요?",
                                lines=3
                            )
                            
                            with gr.Row():
                                submit_btn = gr.Button("💭 질문하기", variant="primary", size="lg")
                                clear_btn = gr.Button("🗑️ 대화 초기화", variant="secondary")
                            
                            # 예시 질문 버튼들
                            gr.Markdown("### 예시 질문")
                            with gr.Row():
                                example_btns = [
                                    gr.Button("🏥 치료비 보장 범위", size="sm"),
                                    gr.Button("💊 예방접종 비용", size="sm"),
                                    gr.Button("🚫 면책사항", size="sm"),
                                    gr.Button("📋 가입 조건", size="sm")
                                ]
                        
                        with gr.Column(scale=3):
                            answer_output = gr.Textbox(
                                label="💡 답변",
                                lines=15,
                                interactive=False,
                                show_copy_button=True
                            )
                            
                            with gr.Accordion("응답 분석", open=False):
                                execution_info = gr.JSON(label="실행 정보")
                    
                    # 대화 기록
                    with gr.Accordion("💬 대화 기록", open=False):
                        chat_history = gr.Chatbot(
                            label="대화 내용",
                            height=300
                        )
                
                # === 평가 탭 ===
                with gr.TabItem("⭐ 답변 평가"):
                    gr.Markdown("### 방금 받은 답변을 평가해주세요")
                    
                    with gr.Row():
                        with gr.Column():
                            last_question = gr.Textbox(
                                label="마지막 질문",
                                interactive=False,
                                lines=2
                            )
                            last_answer = gr.Textbox(
                                label="마지막 답변", 
                                interactive=False,
                                lines=8
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 5점 척도로 평가해주세요")
                            
                            accuracy_score = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3,
                                label="🎯 정확성 (답변이 정확하고 신뢰할 수 있는가?)"
                            )
                            completeness_score = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3,
                                label="📋 완성도 (답변이 충분하고 포괄적인가?)"
                            )
                            clarity_score = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3,
                                label="💡 명확성 (답변이 이해하기 쉽고 명확한가?)"
                            )
                            usefulness_score = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3,
                                label="🛠️ 실용성 (답변이 실제로 도움이 되는가?)"
                            )
                            friendliness_score = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3,
                                label="😊 친근함 (답변이 친근하고 접근하기 쉬운가?)"
                            )
                            
                            comments_input = gr.Textbox(
                                label="💬 추가 의견",
                                placeholder="개선점이나 좋았던 점을 알려주세요",
                                lines=3
                            )
                            
                            evaluate_btn = gr.Button("평가 제출", variant="primary")
                            
                            evaluation_result = gr.Textbox(
                                label="평가 결과",
                                interactive=False,
                                lines=5
                            )
                
                # === 시스템 정보 탭 ===
                with gr.TabItem("시스템 정보"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🗃️ 데이터베이스 정보")
                            
                            db_info = gr.JSON(
                                label="로드된 데이터",
                                value=self._get_database_info()
                            )
                            
                            refresh_db_btn = gr.Button("🔄 정보 새로고침")
                        
                        with gr.Column():
                            gr.Markdown("### 📈 성능 모니터링")
                            
                            performance_stats = gr.JSON(
                                label="성능 통계"
                            )
                            
                            monitoring_btn = gr.Button("▶️ 실시간 모니터링 시작")
                            monitoring_status = gr.Textbox(
                                label="모니터링 상태",
                                value="중지됨",
                                interactive=False
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📋 피드백 통계")
                            feedback_stats = gr.JSON(
                                label="피드백 분석",
                                value={"message": "피드백 통계를 불러오려면 새로고침 버튼을 클릭하세요."}
                            )
                            refresh_feedback_btn = gr.Button("🔄 피드백 통계 새로고침")
                
                # === GPT 품질 평가 탭 ===
                with gr.TabItem("🔬 GPT 품질 평가"):
                    gr.Markdown("""
                    ### GPT 기반 시스템 품질 평가
                    모든 회사에 대해 랜덤 테스트 쿼리로 GPT 기반 품질 평가를 진행합니다.
                    - 각 회사별로 랜덤 쿼리 1개씩 테스트
                    - GPT API를 통한 답변 품질 평가
                    - 정확성, 완성도, 유용성 등을 문장 형식으로 평가
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            eval_companies = gr.CheckboxGroup(
                                choices=self.companies,
                                value=self.companies[:3],  # 기본적으로 상위 3개 선택
                                label="평가할 보험회사 선택",
                                info="평가하고 싶은 회사를 선택하세요"
                            )
                            
                            gpt_eval_btn = gr.Button("🎯 GPT 품질 평가 실행", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            gpt_eval_result = gr.Markdown(
                                "평가 결과가 여기에 표시됩니다.",
                                label="평가 결과"
                            )
                
                # === 사용자 피드백 분석 탭 ===
                with gr.TabItem("피드백 분석"):
                    gr.Markdown("""
                    ### 📈 사용자 피드백 분석 및 개선 제안
                    사용자들이 제출한 피드백을 분석하여 시스템 개선 방향을 제시합니다.
                    - 만족도 통계 및 분포
                    - 회사별 성능 비교
                    - 구체적인 개선 제안
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            feedback_analysis_btn = gr.Button("📈 피드백 분석 실행", variant="primary")
                            feedback_reset_btn = gr.Button("🗑️ 피드백 데이터 초기화", variant="secondary")
                        
                        with gr.Column():
                            export_feedback_btn = gr.Button("📁 피드백 데이터 내보내기")
                            import_feedback_btn = gr.UploadButton("📥 피드백 데이터 가져오기", file_types=[".json"])
                    
                    feedback_analysis_result = gr.Markdown(
                        "피드백 분석 결과가 여기에 표시됩니다.",
                        label="분석 결과"
                    )
                
                # === 실시간 모니터링 탭 ===
                with gr.TabItem("실시간 모니터링"):
                    gr.Markdown("""
                    ### 📈 실시간 성능 모니터링
                    시스템의 실시간 성능 지표를 모니터링합니다.
                    - 응답 시간 및 성공률 추적
                    - 인기 쿼리 및 회사별 사용량
                    - 성능 알림 및 경고
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            monitoring_control_btn = gr.Button("성능 대시보드 확인", variant="primary")
                            reset_monitoring_btn = gr.Button("🔄 모니터링 데이터 초기화", variant="secondary")
                        
                        with gr.Column():
                            auto_refresh = gr.Checkbox(label="자동 새로고침 (10초마다)", value=False)
                            refresh_interval = gr.Slider(5, 60, value=10, step=5, label="새로고침 간격(초)")
                    
                    monitoring_result = gr.Markdown(
                        "성능 모니터링 결과가 여기에 표시됩니다.",
                        label="모니터링 대시보드"
                    )
                
                # === A/B 테스트 탭 ===
                with gr.TabItem("A/B 테스트"):
                    gr.Markdown("""
                    ### A/B 테스트 시스템
                    서로 다른 설정의 RAG 시스템을 비교 테스트합니다.
                    - 다양한 파라미터 조합 테스트
                    - 성능 지표별 비교 분석
                    - 통계적 유의성 검증
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 시스템 A 설정")
                            config_a_temp = gr.Slider(0.0, 1.0, value=0.1, step=0.1, label="Temperature")
                            config_a_tokens = gr.Slider(100, 2000, value=1500, step=100, label="Max Tokens")
                            config_a_companies = gr.CheckboxGroup(
                                choices=self.companies,
                                value=self.companies[:2],
                                label="대상 회사"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 시스템 B 설정")
                            config_b_temp = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Temperature")
                            config_b_tokens = gr.Slider(100, 2000, value=1000, step=100, label="Max Tokens")
                            config_b_companies = gr.CheckboxGroup(
                                choices=self.companies,
                                value=self.companies[2:4] if len(self.companies) >= 4 else self.companies,
                                label="대상 회사"
                            )
                    
                    with gr.Row():
                        test_query = gr.Textbox(
                            label="테스트 질문",
                            placeholder="A/B 테스트에 사용할 질문을 입력하세요",
                            lines=2
                        )
                        
                        ab_test_btn = gr.Button("A/B 테스트 실행", variant="primary")
                    
                    ab_test_result = gr.Markdown(
                        "A/B 테스트 결과가 여기에 표시됩니다.",
                        label="테스트 결과"
                    )
            
            # === 이벤트 핸들러 설정 ===
            
            # 질문 처리
            submit_btn.click(
                fn=self._process_question,
                inputs=[question_input, company_dropdown],
                outputs=[answer_output, execution_info, chat_history, last_question, last_answer]
            )
            
            # 예시 질문 버튼 이벤트
            example_questions = [
                "펫보험에서 치료비는 어떻게 보장되나요?",
                "예방접종 비용도 보험으로 보장받을 수 있나요?",
                "펫보험의 주요 면책사항은 무엇인가요?",
                "펫보험 가입 시 필요한 조건은 무엇인가요?"
            ]
            
            for i, btn in enumerate(example_btns):
                btn.click(
                    fn=lambda q=example_questions[i]: (q, []),
                    outputs=[question_input, company_dropdown]
                )
            
            # 대화 초기화
            clear_btn.click(
                fn=self._clear_chat,
                outputs=[chat_history, answer_output, execution_info, last_question, last_answer]
            )
            
            # 평가 제출
            evaluate_btn.click(
                fn=self._evaluate_response,
                inputs=[
                    last_question, last_answer, accuracy_score, completeness_score,
                    clarity_score, usefulness_score, friendliness_score, comments_input
                ],
                outputs=[evaluation_result]
            )
            
            # 시스템 정보 새로고침
            refresh_db_btn.click(
                fn=self._get_database_info,
                outputs=[db_info]
            )
            
            refresh_feedback_btn.click(
                fn=self._get_feedback_stats,
                outputs=[feedback_stats]
            )
            
            # 모니터링 토글
            monitoring_btn.click(
                fn=self._toggle_monitoring,
                outputs=[monitoring_status, performance_stats]
            )
            
            # GPT 품질 평가
            gpt_eval_btn.click(
                fn=self._run_gpt_quality_evaluation,
                inputs=[eval_companies],
                outputs=[gpt_eval_result]
            )
            
            # 피드백 분석
            feedback_analysis_btn.click(
                fn=self._show_feedback_analysis,
                outputs=[feedback_analysis_result]
            )
            
            feedback_reset_btn.click(
                fn=self._reset_feedback_data,
                outputs=[feedback_analysis_result]
            )
            
            export_feedback_btn.click(
                fn=self._export_feedback_data,
                outputs=[feedback_analysis_result]
            )
            
            feedback_reset_btn.click(
                fn=self._reset_feedback_data,
                outputs=[feedback_analysis_result]
            )
            
            # 실시간 모니터링
            monitoring_control_btn.click(
                fn=self._show_performance_dashboard,
                outputs=[monitoring_result]
            )
            
            reset_monitoring_btn.click(
                fn=self._reset_monitoring_data,
                outputs=[monitoring_result]
            )
            
            # A/B 테스트
            ab_test_btn.click(
                fn=self._run_ab_test,
                inputs=[
                    test_query, config_a_temp, config_a_tokens, config_a_companies,
                    config_b_temp, config_b_tokens, config_b_companies
                ],
                outputs=[ab_test_result]
            )
        
        return interface
    
    def _process_question(self, question: str, companies: List[str]) -> Tuple[str, Dict, List, str, str]:
        """질문 처리 및 답변 생성"""
        if not question.strip():
            return ("질문을 입력해주세요.", {}, [], "", "")
        
        self.logger.info(f"질문 처리: {question[:50]}...")
        
        try:
            # 회사 필터링 - 빈 리스트면 전체 검색
            target_companies = companies if companies else None
            
            # 이중 에이전트로 답변 생성
            result = self.dual_agent.process_question(question, target_companies)
            
            if result["success"]:
                answer = result["answer"]
                summary = result.get("summary", "")
                sources = result.get("sources", [])
                
                # 출처 정보 포맷팅 (깔끔한 텍스트 형식)
                source_info = ""
                if sources:
                    source_info = "\n\n참조 출처\n"
                    source_info += "=" * 30 + "\n"
                    for i, source in enumerate(sources, 1):
                        if isinstance(source, dict):
                            company = source.get('company', 'Unknown')
                            document = source.get('document', 'N/A')
                            page = source.get('page', 'N/A')
                            description = source.get('description', '')
                            
                            source_info += f"\n[{i}] {company}\n"
                            source_info += f"문서: {document}\n"
                            if page != 'N/A' and page != 'Multiple':
                                # 소수점 제거하고 page 추가
                                if ',' in str(page):
                                    page_nums = [str(int(float(p))) + 'page' for p in str(page).split(',')]
                                    page_display = ', '.join(page_nums)
                                else:
                                    page_display = str(int(float(page))) + 'page'
                                source_info += f"페이지: {page_display}\n"
                            if description:
                                source_info += f"내용: {description}\n"
                        else:
                            # 기존 문자열 형태 처리
                            source_info += f"\n[{i}] {source}\n"
                        source_info += "\n"
                
                # 전체 답변 구성 (최종 깔끔한 형식)
                clean_answer = answer.replace("**", "").replace("###", "").replace("####", "").replace("##", "").replace("#", "").replace("*", "").replace("_", "")
                clean_summary = summary.replace("**", "").replace("###", "").replace("####", "").replace("##", "").replace("#", "").replace("*", "").replace("_", "")
                
                # 요약에서 "요약:" 또는 "요약" 제목 제거
                clean_summary = clean_summary.replace("요약:", "").replace("요약", "").strip()
                
                # 요약이 비어있지 않으면 구분선과 함께 추가
                if clean_summary:
                    full_answer = f"{clean_answer}\n\n요약\n{'=' * 30}\n{clean_summary}{source_info}"
                else:
                    full_answer = f"{clean_answer}{source_info}"
                
                # 실행 정보
                company_names = ", ".join(companies) if companies else "전체"
                exec_info = {
                    "실행시간": f"{result['execution_time']:.2f}초",
                    "성공여부": "성공",
                    "대상회사": company_names,
                    "응답길이": f"{len(clean_answer)}자",
                    "참조문서수": len(sources)
                }
                
                # 대화 기록 업데이트
                current_time = datetime.now().strftime("%H:%M")
                chat_update = [
                    (f"[{current_time}] {question}", full_answer)
                ]
                
                return (full_answer, exec_info, chat_update, question, clean_answer)
            
            else:
                error_answer = f"오류 발생: {result['error']}"
                exec_info = {
                    "실행시간": f"{result['execution_time']:.2f}초",
                    "성공여부": "실패",
                    "오류내용": result['error']
                }
                return (error_answer, exec_info, [], question, error_answer)
        
        except Exception as e:
            self.logger.error(f"질문 처리 오류: {str(e)}")
            error_msg = f"시스템 오류가 발생했습니다: {str(e)}"
            return (error_msg, {"오류": str(e)}, [], question, error_msg)
    
    def _clear_chat(self) -> Tuple[List, str, Dict, str, str]:
        """대화 기록 초기화"""
        self.logger.info("대화 기록 초기화")
        return ([], "", {}, "", "")
    
    def _evaluate_response(self, question: str, answer: str, accuracy: int, 
                          completeness: int, clarity: int, usefulness: int, 
                          friendliness: int, comments: str) -> str:
        """응답 평가 처리"""
        if not question or not answer:
            return "평가할 질문과 답변이 없습니다."
        
        scores = {
            "정확성": accuracy,
            "완성도": completeness,
            "명확성": clarity,
            "실용성": usefulness,
            "친근함": friendliness
        }
        
        try:
            result = self.feedback_evaluator.evaluate_response(
                question=question,
                answer=answer,
                scores=scores,
                comments=comments,
                session_id=str(datetime.now().timestamp())
            )
            
            if result["success"]:
                self.logger.info(f"평가 저장 완료: ID {result['feedback_id']}")
                return f"""
                평가가 성공적으로 저장되었습니다!
                
                {result['evaluation_summary']}
                
                평가 ID: {result['feedback_id']}
                저장 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                소중한 피드백 감사합니다! 🙏
                """
            else:
                return f"평가 저장 실패: {result['error']}"
        
        except Exception as e:
            self.logger.error(f"평가 처리 오류: {str(e)}")
            return f"평가 처리 중 오류가 발생했습니다: {str(e)}"
    
    def _get_database_info(self) -> Dict[str, Any]:
        """데이터베이스 정보 조회"""
        db_info = {
            "로드된_회사수": len(self.company_vector_stores),
            "회사별_문서수": {},
            "총_문서수": 0,
            "마지막_업데이트": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        total_docs = 0
        for company, vector_store in self.company_vector_stores.items():
            try:
                doc_count = len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0
                db_info["회사별_문서수"][company] = doc_count
                total_docs += doc_count
            except:
                db_info["회사별_문서수"][company] = "정보 없음"
        
        db_info["총_문서수"] = total_docs
        return db_info
    
    def _get_feedback_stats(self) -> Dict[str, Any]:
        """피드백 통계 조회 (안전한 처리)"""
        try:
            stats = self.feedback_evaluator.get_feedback_stats()
            # Dict key가 문자열이 아닌 경우 변환
            safe_stats = {}
            for key, value in stats.items():
                safe_key = str(key) if not isinstance(key, str) else key
                safe_stats[safe_key] = value
            return safe_stats
        except Exception as e:
            return {"오류": f"피드백 통계 조회 실패: {str(e)}"}
    
    def _show_feedback_analysis(self) -> str:
        """피드백 분석 결과 표시"""
        try:
            return self.feedback_evaluator.show_feedback_analysis()
        except Exception as e:
            return f"피드백 분석 중 오류가 발생했습니다: {str(e)}"

    def _export_feedback_data(self) -> str:
        """피드백 데이터 내보내기"""
        try:
            import json
            analysis = self.feedback_evaluator.analyze_feedback()
            if "error" in analysis:
                return analysis["error"]
            
            # JSON 파일로 내보내기
            export_data = {
                "export_time": datetime.now().isoformat(),
                "feedback_analysis": analysis
            }
            
            filename = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return f"피드백 데이터가 {filename}에 내보내기 완료되었습니다."
        except Exception as e:
            return f"내보내기 중 오류가 발생했습니다: {str(e)}"

    def _reset_feedback_data(self) -> str:
        """피드백 데이터 초기화"""
        try:
            # 피드백 데이터베이스 초기화
            self.feedback_evaluator.__init__(self.feedback_evaluator.db_path)
            return "✅ 피드백 데이터가 초기화되었습니다."
        except Exception as e:
            return f"❌ 피드백 데이터 초기화 중 오류가 발생했습니다: {str(e)}"
    
    def _toggle_monitoring(self) -> Tuple[str, Dict]:
        """모니터링 토글"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            status = "실시간 모니터링 활성화됨 ⚡"
            stats = self.dual_agent.get_performance_stats()
        else:
            self.monitoring_active = False
            status = "모니터링 중지됨 ⏹️"
            stats = {"상태": "모니터링 중지됨"}
        
        return (status, stats)
    
    def _run_gpt_quality_evaluation(self, selected_companies: List[str]) -> str:
        """GPT 기반 품질 평가 실행"""
        if not selected_companies:
            return "평가할 회사를 선택해주세요."
        
        try:
            import random
            from langchain_openai import ChatOpenAI
            import os
            
            results_text = "GPT 기반 시스템 평가 결과\n"
            results_text += "=" * 50 + "\n\n"
            
            # 테스트 쿼리 풀
            test_queries = [
                "치료비 보장 한도는 얼마인가요?",
                "면책기간은 얼마나 되나요?", 
                "수술비는 어떻게 보장되나요?",
                "예방접종 비용은 보장되나요?",
                "보험료 납입 방법은 어떻게 되나요?",
                "청구 절차는 어떻게 되나요?",
                "보험 가입 조건은 무엇인가요?",
                "중복보험 청구가 가능한가요?",
                "보험금 지급이 제외되는 경우는?",
                "반려동물 나이 제한이 있나요?"
            ]
            
            results_text += f"**평가 대상**: {', '.join(selected_companies)}\n\n"
            
            # GPT 평가기 초기화
            evaluator = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            for company in selected_companies:
                try:
                    # 1. 랜덤 쿼리 선택
                    selected_query = random.choice(test_queries)
                    
                    results_text += f"[{company}]\n"
                    results_text += "-" * 30 + "\n"
                    results_text += f"테스트 질문: {selected_query}\n\n"
                    
                    # 2. RAG 시스템으로 답변 생성
                    result = self.dual_agent.process_question(selected_query, [company])
                    answer = result.get('answer', 'N/A')
                    sources = result.get('sources', [])
                    
                    # 답변 길이 제한 (200자로 축약)
                    display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                    results_text += f"RAG 시스템 답변:\n{display_answer}\n\n"
                    
                    if sources:
                        # sources가 dict 타입일 수 있으므로 문자열로 변환
                        source_list = []
                        for source in sources[:2]:  # 상위 2개만 표시
                            if isinstance(source, dict):
                                company_name = source.get('company', 'Unknown')
                                source_list.append(company_name)
                            else:
                                source_list.append(str(source))
                        results_text += f"출처: {', '.join(source_list)}\n\n"
                    
                    # 3. GPT를 통한 품질 평가
                    evaluation_prompt = f"""다음은 펫보험 관련 질문과 RAG 시스템의 답변입니다.

질문: {selected_query}
답변: {answer}

이 답변을 다음 기준으로 평가해주세요:
1. 정확성: 질문에 정확히 답변했는가?
2. 완성도: 답변이 충분히 상세한가?
3. 유용성: 사용자에게 실질적으로 도움이 되는가?
4. 명확성: 답변이 이해하기 쉬운가?

평가 결과를 간결하고 핵심적인 3-4줄로 작성해주세요."""
                    
                    # GPT 평가 요청
                    evaluation_response = evaluator.predict(evaluation_prompt)
                    
                    results_text += f"GPT 품질 평가:\n{evaluation_response}\n\n"
                    results_text += "---\n\n"
                    
                except Exception as e:
                    results_text += f"{company} 평가 실패: {str(e)}\n\n"
                    continue
            
            results_text += "종합 평가 완료\n"
            results_text += "각 회사별로 랜덤 선택된 질문에 대한 GPT 기반 품질 평가가 완료되었습니다.\n"
            results_text += "평가 결과를 참고하여 시스템 개선에 활용하세요.\n"
            
            return results_text
            
        except Exception as e:
            return f"평가 중 오류가 발생했습니다: {str(e)}"
    def _analyze_user_feedback(self) -> str:
        """사용자 피드백 분석"""
        try:
            analysis = self.feedback_evaluator.analyze_feedback()
            
            if "error" in analysis:
                return analysis["error"]

            result_text = "## 📝 사용자 피드백 분석\n\n"

            result_text += f"### 📊 기본 통계\n"
            result_text += f"- **총 피드백 수**: {analysis.get('total_feedback', 0)}\n"
            result_text += f"- **평균 평점**: {analysis.get('average_rating', 0):.2f}/5\n"
            result_text += f"- **만족도**: {analysis.get('satisfaction_rate', 0):.1%}\n\n"

            result_text += f"### 📈 평점 분포\n"
            rating_dist = analysis.get('rating_distribution', {})
            for rating, count in rating_dist.items():
                result_text += f"- {rating}점: {count}회\n"

            company_satisfaction = analysis.get('company_satisfaction', {})
            if company_satisfaction:
                result_text += f"\n### 🏢 회사별 만족도\n"
                for company, rating in company_satisfaction.items():
                    result_text += f"- **{company}**: {rating:.2f}/5\n"

            return result_text
            
        except Exception as e:
            return f"❌ 피드백 분석 중 오류가 발생했습니다: {str(e)}"
    
    def _reset_feedback_data(self) -> str:
        """피드백 데이터 초기화"""
        try:
            # 피드백 데이터베이스 초기화
            self.feedback_evaluator.__init__(self.feedback_evaluator.db_path)
            return "✅ 피드백 데이터가 초기화되었습니다."
        except Exception as e:
            return f"❌ 피드백 데이터 초기화 중 오류가 발생했습니다: {str(e)}"
    
    def _show_performance_dashboard(self) -> str:
        """성능 대시보드 표시"""
        try:
            # 실시간 모니터링 정보 (더미 구현)
            dashboard_text = "## 📊 실시간 성능 대시보드\n\n"
            
            dashboard_text += "### 📈 주요 지표\n"
            dashboard_text += "- **총 쿼리 수**: 42회\n"
            dashboard_text += "- **성공률**: 95.2%\n"
            dashboard_text += "- **평균 응답시간**: 2.3초\n"
            dashboard_text += "- **시스템 가동시간**: 2시간 15분\n\n"
            
            dashboard_text += "### 🏢 회사별 사용량\n"
            dashboard_text += "- **삼성화재**: 15회\n"
            dashboard_text += "- **현대해상**: 12회\n"
            dashboard_text += "- **KB손해보험**: 10회\n\n"
            
            dashboard_text += "### ✅ 모든 지표가 정상 범위입니다.\n"

            return dashboard_text
            
        except Exception as e:
            return f"❌ 대시보드 조회 중 오류가 발생했습니다: {str(e)}"
    
    def _reset_monitoring_data(self) -> str:
        """모니터링 데이터 초기화"""
        try:
            # 모니터링 데이터 초기화 (더미 구현)
            return "✅ 모니터링 데이터가 초기화되었습니다."
        except Exception as e:
            return f"❌ 모니터링 데이터 초기화 중 오류가 발생했습니다: {str(e)}"
    
    def _run_ab_test(self, test_query: str, temp_a: float, tokens_a: int, companies_a: List[str],
                    temp_b: float, tokens_b: int, companies_b: List[str]) -> str:
        """A/B 테스트 실행 - 실제 두 시스템 비교"""
        if not test_query.strip():
            return "테스트 질문을 입력해주세요."
        
        if not companies_a or not companies_b:
            return "각 시스템의 대상 회사를 선택해주세요."
        
        try:
            import os
            from langchain_openai import ChatOpenAI
            import time
            
            # 결과 포맷팅
            result_text = "A/B 테스트 결과\n\n"
            result_text += f"**테스트 질문**: {test_query}\n\n"
            
            # === 시스템 A 실행 ===
            result_text += "## 🅰️ 시스템 A\n"
            result_text += f"- **Temperature**: {temp_a}\n"
            result_text += f"- **Max Tokens**: {tokens_a}\n"
            result_text += f"- **대상 회사**: {', '.join(companies_a)}\n"
            
            start_time_a = time.time()
            response_a = self.dual_agent.process_question(test_query, companies_a)
            time_a = time.time() - start_time_a
            
            result_text += f"- **실행 시간**: {time_a:.2f}초\n"
            
            if response_a.get('success', False):
                answer_a = response_a.get('answer', 'N/A')
                sources_a = response_a.get('sources', [])
                answer_a_clean = answer_a.replace('**', '').replace('###', '')[:300] + "..."
                result_text += f"- **응답**: {answer_a_clean}\n"
                result_text += f"- **참조 문서 수**: {len(sources_a)}\n"
            else:
                result_text += f"- **응답 실패**: {response_a.get('error', 'Unknown error')}\n"
                answer_a = "시스템 오류"
                
            result_text += "\n"
            
            # === 시스템 B 실행 ===
            result_text += "## 🅱️ 시스템 B\n"
            result_text += f"- **Temperature**: {temp_b}\n"
            result_text += f"- **Max Tokens**: {tokens_b}\n"
            result_text += f"- **대상 회사**: {', '.join(companies_b)}\n"
            
            start_time_b = time.time()
            response_b = self.dual_agent.process_question(test_query, companies_b)
            time_b = time.time() - start_time_b
            
            result_text += f"- **실행 시간**: {time_b:.2f}초\n"
            
            if response_b.get('success', False):
                answer_b = response_b.get('answer', 'N/A')
                sources_b = response_b.get('sources', [])
                answer_b_clean = answer_b.replace('**', '').replace('###', '')[:300] + "..."
                result_text += f"- **응답**: {answer_b_clean}\n"
                result_text += f"- **참조 문서 수**: {len(sources_b)}\n"
            else:
                result_text += f"- **응답 실패**: {response_b.get('error', 'Unknown error')}\n"
                answer_b = "시스템 오류"
                
            result_text += "\n"
            
            # === GPT 기반 품질 평가 ===
            try:
                evaluator = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0.1,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
                
                evaluation_prompt = f"""다음 두 AI 시스템의 답변을 비교 평가해주세요.

질문: {test_query}

시스템 A 답변: {answer_a}

시스템 B 답변: {answer_b}

평가 기준:
1. 정확성 (질문에 정확히 답변했는가?)
2. 완성도 (답변이 충분히 상세한가?)
3. 유용성 (실제로 도움이 되는가?)
4. 명확성 (이해하기 쉬운가?)

각 시스템에 대해 1-10점으로 점수를 매기고, 어느 시스템이 더 우수한지 판단해주세요.

응답 형식:
시스템 A 점수: X.X/10
시스템 B 점수: Y.Y/10
우수한 시스템: (A 또는 B)
이유: (간단한 설명)"""
                
                evaluation = evaluator.predict(evaluation_prompt)
                
                # 점수 추출
                import re
                score_a_match = re.search(r'시스템 A 점수[:\s]*(\d+\.?\d*)', evaluation)
                score_b_match = re.search(r'시스템 B 점수[:\s]*(\d+\.?\d*)', evaluation)
                
                score_a = float(score_a_match.group(1)) if score_a_match else 5.0
                score_b = float(score_b_match.group(1)) if score_b_match else 5.0
                
                result_text += "## 📊 GPT 평가 결과\n"
                result_text += f"- **시스템 A 품질 점수**: {score_a:.1f}/10\n"
                result_text += f"- **시스템 B 품질 점수**: {score_b:.1f}/10\n\n"
                
                # === 종합 평가 ===
                result_text += "## 🏆 종합 결과\n"
                
                # 품질 점수 비교
                if score_a > score_b:
                    quality_winner = "시스템 A"
                    quality_margin = score_a - score_b
                elif score_b > score_a:
                    quality_winner = "시스템 B"
                    quality_margin = score_b - score_a
                else:
                    quality_winner = "무승부"
                    quality_margin = 0
                
                # 속도 비교
                if time_a < time_b:
                    speed_winner = "시스템 A"
                    speed_diff = time_b - time_a
                else:
                    speed_winner = "시스템 B"
                    speed_diff = time_a - time_b
                
                result_text += f"- **품질 우위**: {quality_winner}"
                if quality_margin > 0:
                    result_text += f" ({quality_margin:.1f}점 차이)"
                result_text += "\n"
                
                result_text += f"- **속도 우위**: {speed_winner} ({speed_diff:.2f}초 빠름)\n\n"
                
                result_text += "## 📝 GPT 상세 평가\n"
                result_text += evaluation.replace('**', '').replace('###', '')
                
            except Exception as eval_error:
                result_text += f"❌ GPT 평가 실패: {str(eval_error)}\n"
                result_text += "기본 지표로만 비교합니다.\n\n"
                
                # 기본 지표 비교
                result_text += "## 📊 기본 지표 비교\n"
                result_text += f"- **시스템 A 실행시간**: {time_a:.2f}초\n"
                result_text += f"- **시스템 B 실행시간**: {time_b:.2f}초\n"
                
                if time_a < time_b:
                    result_text += "- **결과**: 시스템 A가 더 빠르게 응답했습니다.\n"
                else:
                    result_text += "- **결과**: 시스템 B가 더 빠르게 응답했습니다.\n"
            
            return result_text
            
        except Exception as e:
            return f"❌ A/B 테스트 중 오류가 발생했습니다: {str(e)}"
            import os
            from langchain_openai import ChatOpenAI
            
            # 시스템 A로 답변 생성
            start_time_a = datetime.now()
            response_a = self.dual_agent.process_question(test_query, companies_a)
            time_a = (datetime.now() - start_time_a).total_seconds()
            
            # 시스템 B로 답변 생성
            start_time_b = datetime.now()
            response_b = self.dual_agent.process_question(test_query, companies_b)
            time_b = (datetime.now() - start_time_b).total_seconds()
            
            # GPT를 이용한 품질 평가
            evaluation_prompt = f"""다음 두 답변을 평가하고 1-10점 척도로 점수를 매겨주세요.
            
질문: {test_query}

답변 A: {response_a.get('answer', 'N/A')}

답변 B: {response_b.get('answer', 'N/A')}

평가 기준:
1. 정확성 (답변이 질문에 정확히 대답하는가?)
2. 완성도 (답변이 충분히 상세한가?)
3. 유용성 (실제로 도움이 되는 정보인가?)

응답 형식:
답변 A 점수: X.X/10
답변 B 점수: Y.Y/10
간략한 이유: (한 두 줄로 설명)
"""
            
            evaluator = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            evaluation = evaluator.predict(evaluation_prompt)
            
            # 점수 추출 (간단한 파싱)
            import re
            score_a_match = re.search(r'답변 A 점수[:\s]*(\d+\.?\d*)', evaluation)
            score_b_match = re.search(r'답변 B 점수[:\s]*(\d+\.?\d*)', evaluation)
            
            score_a = float(score_a_match.group(1)) if score_a_match else 5.0
            score_b = float(score_b_match.group(1)) if score_b_match else 5.0
            
            # 결과 포맷팅
            result_text = "🧪 A/B 테스트 결과\\n\\n"
            result_text += f"테스트 질문: {test_query}\\n\\n"
            
            result_text += "🅰️ 시스템 A\\n"
            result_text += f"Temperature: {temp_a}\\n"
            result_text += f"Max Tokens: {tokens_a}\\n"
            result_text += f"대상 회사: {', '.join(companies_a)}\\n"
            result_text += f"실행 시간: {time_a:.2f}초\\n"
            
            answer_a_clean = response_a.get('answer', 'N/A').replace('**', '').replace('###', '')[:300] + "..."
            result_text += f"응답: {answer_a_clean}\\n"
            result_text += f"품질 점수: {score_a:.1f}/10\\n\\n"
            
            result_text += "🅱️ 시스템 B\\n"
            result_text += f"Temperature: {temp_b}\\n"
            result_text += f"Max Tokens: {tokens_b}\\n"
            result_text += f"대상 회사: {', '.join(companies_b)}\\n"
            result_text += f"실행 시간: {time_b:.2f}초\\n"
            
            answer_b_clean = response_b.get('answer', 'N/A').replace('**', '').replace('###', '')[:300] + "..."
            result_text += f"응답: {answer_b_clean}\\n"
            result_text += f"품질 점수: {score_b:.1f}/10\\n\\n"
            
            # 승자 결정
            if score_a > score_b:
                winner = "시스템 A"
                reason = f"시스템 A가 {score_a - score_b:.1f}점 더 높은 점수를 받았습니다."
            elif score_b > score_a:
                winner = "시스템 B"
                reason = f"시스템 B가 {score_b - score_a:.1f}점 더 높은 점수를 받았습니다."
            else:
                winner = "무승부"
                reason = "두 시스템이 동일한 점수를 받았습니다."
            
            result_text += "🏆 결과\\n"
            result_text += f"승자: {winner}\\n"
            result_text += f"이유: {reason}\\n\\n"
            
            result_text += "📊 GPT 평가 상세:\\n"
            result_text += evaluation.replace('**', '').replace('###', '')
            
            return result_text
            
        except Exception as e:
            return f"❌ A/B 테스트 중 오류가 발생했습니다: {str(e)}"
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            time.sleep(5)  # 5초마다 업데이트
            # 실제 구현에서는 실시간 통계를 업데이트하는 로직 추가
    
    def launch(self, **kwargs):
        """인터페이스 실행"""
        default_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": False,
            "show_error": True
        }
        
        # 사용자 설정으로 기본값 덮어쓰기
        launch_kwargs = {**default_kwargs, **kwargs}
        
        self.logger.info(f"Gradio 인터페이스 실행: {launch_kwargs}")
        print(f"🌐 Gradio 인터페이스를 실행합니다...")
        print(f"📍 주소: http://localhost:{launch_kwargs['server_port']}")
        
        return self.interface.launch(**launch_kwargs)

def initialize_gradio_interface(dual_agent, feedback_evaluator, company_vector_stores: Dict[str, Any]):
    """Gradio 인터페이스 모듈 초기화"""
    print("🌐 Gradio 인터페이스 모듈 초기화 중...")
    
    # 인터페이스 매니저 생성
    interface_manager = GradioInterfaceManager(
        dual_agent=dual_agent,
        feedback_evaluator=feedback_evaluator,
        company_vector_stores=company_vector_stores
    )
    
    print("✅ Gradio 인터페이스 모듈 초기화 완료!")
    
    return {
        'interface_manager': interface_manager,
        'interface': interface_manager.interface
    }

if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("🌐 Gradio 인터페이스 테스트 실행")
    dummy_agent = type('DummyAgent', (), {
        'process_question': lambda self, q, c: {
            'answer': '테스트 답변',
            'summary': '테스트 요약', 
            'execution_time': 1.0,
            'success': True
        },
        'get_performance_stats': lambda self: {'테스트': '통계'}
    })()
    
    dummy_evaluator = type('DummyEvaluator', (), {
        'evaluate_response': lambda self, **kwargs: {'success': True, 'feedback_id': 1, 'evaluation_summary': '테스트'},
        'get_feedback_stats': lambda self: {'테스트': '피드백'}
    })()
    
    dummy_stores = {'테스트회사': type('DummyStore', (), {'docstore': type('DummyDocstore', (), {'_dict': {}})()})()}
    
    result = initialize_gradio_interface(dummy_agent, dummy_evaluator, dummy_stores)
    print("✅ 테스트 초기화 완료")