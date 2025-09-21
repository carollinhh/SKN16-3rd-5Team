"""
펫보험 RAG 시스템 평가 모듈
- 사용자 피드백 수집 및 분석
- A/B 테스트 시스템
- 실시간 성능 모니터링
- GPT 기반 품질 평가
"""

import sqlite3
import json
import statistics
import threading
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from pathlib import Path


class UserFeedbackEvaluator:
    """사용자 피드백 수집 및 분석 시스템"""

    def __init__(self, db_path: str = "user_feedback.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            company TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER NOT NULL,
            feedback_text TEXT,
            sources TEXT
        )
        ''')

        conn.commit()
        conn.close()

    def collect_feedback(self, query: str, company: str, answer: str, sources: list,
                        rating: int, feedback_text: str = ""):
        """피드백 수집"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 테이블이 존재하는지 확인하고 스키마를 업데이트
        cursor.execute("PRAGMA table_info(feedback)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'query' not in columns:
            # 기존 테이블 삭제하고 새로 생성
            cursor.execute("DROP TABLE IF EXISTS feedback")
            self.init_database()

        sources_str = json.dumps(sources, ensure_ascii=False) if sources else ""
        timestamp = datetime.now().isoformat()

        cursor.execute('''
        INSERT INTO feedback (timestamp, query, company, answer, rating, feedback_text, sources)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, query, company, answer, rating, feedback_text, sources_str))

        conn.commit()
        conn.close()

        print(f"✅ 피드백 저장 완료: {company} - 평점 {rating}/5")

    def analyze_feedback(self) -> Dict[str, Any]:
        """피드백 분석"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM feedback')
        all_feedback = cursor.fetchall()

        if not all_feedback:
            return {"error": "수집된 피드백이 없습니다."}

        # 컬럼 이름 매핑
        columns = [desc[0] for desc in cursor.description]
        feedback_data = [dict(zip(columns, row)) for row in all_feedback]

        conn.close()

        # 분석 결과
        ratings = [f['rating'] for f in feedback_data]
        companies = [f['company'] for f in feedback_data]

        analysis = {
            'total_feedback': len(feedback_data),
            'average_rating': statistics.mean(ratings),
            'rating_distribution': Counter(ratings),
            'company_feedback_count': Counter(companies),
            'satisfaction_rate': len([r for r in ratings if r >= 4]) / len(ratings),
            'recent_feedback': feedback_data[-5:] if len(feedback_data) >= 5 else feedback_data
        }

        # 회사별 평균 평점
        company_ratings = defaultdict(list)
        for f in feedback_data:
            company_ratings[f['company']].append(f['rating'])

        analysis['company_satisfaction'] = {
            company: statistics.mean(ratings)
            for company, ratings in company_ratings.items()
        }

        return analysis

    def generate_improvement_suggestions(self) -> List[str]:
        """개선 제안 생성"""
        analysis = self.analyze_feedback()

        if "error" in analysis:
            return ["피드백 데이터가 없어 개선 제안을 생성할 수 없습니다."]

        suggestions = []

        # 평균 평점 기반 제안
        avg_rating = analysis['average_rating']
        if avg_rating < 3.0:
            suggestions.append("전반적인 답변 품질 개선이 필요합니다. 답변의 정확성과 완성도를 높여주세요.")
        elif avg_rating < 3.5:
            suggestions.append("답변 품질이 보통 수준입니다. 더 구체적이고 유용한 정보 제공을 고려해주세요.")

        # 회사별 성능 차이 분석
        company_satisfaction = analysis.get('company_satisfaction', {})
        if company_satisfaction:
            best_company = max(company_satisfaction.items(), key=lambda x: x[1])
            worst_company = min(company_satisfaction.items(), key=lambda x: x[1])

            if best_company[1] - worst_company[1] > 1.0:
                suggestions.append(f"{worst_company[0]}의 답변 품질을 {best_company[0]} 수준으로 개선이 필요합니다.")

        # 평점 분포 분석
        rating_dist = analysis['rating_distribution']
        low_ratings = rating_dist.get(1, 0) + rating_dist.get(2, 0)
        total_feedback = analysis['total_feedback']

        if low_ratings / total_feedback > 0.2:
            suggestions.append("낮은 평점(1-2점)의 비율이 높습니다. 답변 정확성과 관련성을 검토해주세요.")

        if not suggestions:
            suggestions.append("현재 피드백 점수가 양호합니다. 지속적인 품질 유지에 노력해주세요.")

        return suggestions

    def evaluate_response(self, question: str, answer: str, scores: Dict[str, int], 
                         comments: str = "", session_id: str = "", company: str = "") -> Dict[str, Any]:
        """응답 평가 및 저장 (Gradio 인터페이스용)"""
        
        # 점수 검증
        for criterion, score in scores.items():
            if not (1 <= score <= 5):
                return {
                    "success": False,
                    "error": f"{criterion} 점수는 1-5 범위여야 합니다."
                }
        
        # 전체 평균 점수 계산
        overall_score = sum(scores.values()) / len(scores)
        
        try:
            # 기존 collect_feedback 메서드 사용
            self.collect_feedback(
                query=question,
                company=company or "전체",
                answer=answer,
                sources=[],
                rating=int(overall_score),
                feedback_text=comments
            )
            
            return {
                "success": True,
                "feedback_id": "saved",
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
        
        summary = f"전체 평점: {overall_score:.1f}/5.0\\n"
        
        if high_scores:
            summary += f"강점: {', '.join(high_scores)}\\n"
        
        if low_scores:
            summary += f"개선 필요: {', '.join(low_scores)}\\n"
        
        return summary
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """피드백 통계 조회 (Gradio 인터페이스용) - 안전한 Dict key 처리"""
        try:
            analysis = self.analyze_feedback()
            if "error" in analysis:
                return analysis
            
            # Gradio 표시용으로 포맷팅 - 모든 key를 문자열로 보장
            stats = {
                "총_피드백_수": analysis.get('total_feedback', 0),
                "평균_평점": f"{analysis.get('average_rating', 0):.2f}/5.0",
                "만족도": f"{analysis.get('satisfaction_rate', 0):.1%}",
                "회사별_평점": self._safe_dict_keys(analysis.get('company_satisfaction', {})),
                "평점_분포": self._safe_dict_keys(analysis.get('rating_distribution', {})),
                "최근_피드백": analysis.get('recent_feedback', [])
            }
            
            return stats
        except Exception as e:
            return {"오류": f"피드백 통계 조회 실패: {str(e)}"}
    
    def _safe_dict_keys(self, d: Dict) -> Dict[str, Any]:
        """Dict의 모든 key를 문자열로 변환하여 안전하게 처리"""
        if not isinstance(d, dict):
            return {}
        return {str(k): v for k, v in d.items()}

    def show_feedback_analysis(self) -> str:
        """피드백 분석 결과를 Gradio용 마크다운 형식으로 반환"""
        analysis = self.analyze_feedback()
        suggestions = self.generate_improvement_suggestions()

        if "error" in analysis:
            return analysis["error"]

        result_text = "# 📝 사용자 피드백 분석\n\n"

        result_text += f"## 📊 기본 통계\n"
        result_text += f"- **총 피드백 수**: {analysis['total_feedback']}\n"
        result_text += f"- **평균 평점**: {analysis['average_rating']:.2f}/5\n"
        result_text += f"- **만족도**: {analysis['satisfaction_rate']:.1%}\n\n"

        result_text += f"## 📈 평점 분포\n"
        for rating, count in analysis['rating_distribution'].items():
            result_text += f"- **{rating}점**: {count}회\n"

        if analysis.get('company_satisfaction'):
            result_text += f"\n## 🏢 회사별 만족도\n"
            for company, rating in analysis['company_satisfaction'].items():
                result_text += f"- **{company}**: {rating:.2f}/5\n"

        if suggestions:
            result_text += f"\n## 💡 개선 제안\n"
            for suggestion in suggestions:
                result_text += f"- {suggestion}\n"

        return result_text


class ABTestEvaluator:
    """A/B 테스트 평가 시스템"""

    def __init__(self):
        self.test_configs = {}
        self.test_results = {}

    def setup_ab_test(self, test_name: str, config_a: dict, config_b: dict):
        """A/B 테스트 설정"""
        self.test_configs[test_name] = {
            'config_a': config_a,
            'config_b': config_b,
            'results_a': [],
            'results_b': [],
            'start_time': datetime.now()
        }
        print(f"✅ A/B 테스트 '{test_name}' 설정 완료")

    def record_result(self, test_name: str, variant: str, metrics: dict):
        """결과 기록"""
        if test_name not in self.test_configs:
            print(f"❌ 테스트 '{test_name}'을 찾을 수 없습니다.")
            return

        if variant == 'A':
            self.test_configs[test_name]['results_a'].append(metrics)
        elif variant == 'B':
            self.test_configs[test_name]['results_b'].append(metrics)
        else:
            print(f"❌ 잘못된 variant: {variant}")

    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """A/B 테스트 분석"""
        if test_name not in self.test_configs:
            return {"error": f"테스트 '{test_name}'을 찾을 수 없습니다."}

        test_data = self.test_configs[test_name]
        results_a = test_data['results_a']
        results_b = test_data['results_b']

        if not results_a or not results_b:
            return {"error": "충분한 데이터가 없습니다."}

        # 주요 메트릭 비교
        analysis = {
            'test_name': test_name,
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b),
            'duration': (datetime.now() - test_data['start_time']).days
        }

        # 응답 시간 비교
        if all('response_time' in r for r in results_a + results_b):
            avg_time_a = statistics.mean([r['response_time'] for r in results_a])
            avg_time_b = statistics.mean([r['response_time'] for r in results_b])

            analysis['avg_response_time'] = {
                'variant_a': avg_time_a,
                'variant_b': avg_time_b,
                'improvement': ((avg_time_a - avg_time_b) / avg_time_a) * 100
            }

        # 성공률 비교
        if all('success' in r for r in results_a + results_b):
            success_rate_a = sum(r['success'] for r in results_a) / len(results_a)
            success_rate_b = sum(r['success'] for r in results_b) / len(results_b)

            analysis['success_rate'] = {
                'variant_a': success_rate_a,
                'variant_b': success_rate_b,
                'improvement': ((success_rate_b - success_rate_a) / success_rate_a) * 100
            }

        return analysis


class RealTimeMonitor:
    """실시간 모니터링 시스템"""

    def __init__(self):
        self.metrics = {
            'query_count': 0,
            'success_count': 0,
            'error_count': 0,
            'total_response_time': 0.0,
            'company_usage': defaultdict(int),
            'popular_queries': defaultdict(int),
            'recent_errors': [],
            'hourly_stats': defaultdict(lambda: {'queries': 0, 'errors': 0, 'response_time': 0.0})
        }
        self.lock = threading.Lock()
        self.start_time = datetime.now()

    def log_query(self, query: str, company: str, response_time: float,
                  success: bool, error_type: str = None):
        """쿼리 로깅"""
        with self.lock:
            current_hour = datetime.now().strftime('%Y-%m-%d %H:00')

            self.metrics['query_count'] += 1
            self.metrics['total_response_time'] += response_time
            self.metrics['company_usage'][company] += 1
            self.metrics['popular_queries'][query] += 1

            # 시간별 통계
            self.metrics['hourly_stats'][current_hour]['queries'] += 1
            self.metrics['hourly_stats'][current_hour]['response_time'] += response_time

            if success:
                self.metrics['success_count'] += 1
            else:
                self.metrics['error_count'] += 1
                self.metrics['hourly_stats'][current_hour]['errors'] += 1

                # 최근 에러 기록 (최대 10개)
                error_info = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'company': company,
                    'error_type': error_type or 'Unknown'
                }
                self.metrics['recent_errors'].append(error_info)
                if len(self.metrics['recent_errors']) > 10:
                    self.metrics['recent_errors'].pop(0)

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        with self.lock:
            if self.metrics['query_count'] == 0:
                return {"message": "아직 쿼리가 처리되지 않았습니다."}

            avg_response_time = self.metrics['total_response_time'] / self.metrics['query_count']
            success_rate = (self.metrics['success_count'] / self.metrics['query_count']) * 100
            error_rate = (self.metrics['error_count'] / self.metrics['query_count']) * 100

            # 인기 회사 및 쿼리
            top_companies = dict(Counter(self.metrics['company_usage']).most_common(3))
            top_queries = dict(Counter(self.metrics['popular_queries']).most_common(3))

            uptime = datetime.now() - self.start_time

            summary = {
                'total_queries': self.metrics['query_count'],
                'success_rate': f"{success_rate:.1f}%",
                'error_rate': f"{error_rate:.1f}%",
                'avg_response_time': f"{avg_response_time:.2f}초",
                'uptime': str(uptime).split('.')[0],  # 마이크로초 제거
                'top_companies': top_companies,
                'top_queries': top_queries
            }

            return summary

    def get_alerts(self) -> List[str]:
        """경고 메시지 반환"""
        alerts = []

        if self.metrics['query_count'] > 0:
            error_rate = (self.metrics['error_count'] / self.metrics['query_count']) * 100
            avg_response_time = self.metrics['total_response_time'] / self.metrics['query_count']

            if error_rate > 20:
                alerts.append(f"⚠️ 높은 에러율: {error_rate:.1f}% (임계값: 20%)")

            if avg_response_time > 10:
                alerts.append(f"⚠️ 느린 응답 시간: {avg_response_time:.2f}초 (임계값: 10초)")

            # 최근 에러 확인
            recent_errors = [e for e in self.metrics['recent_errors']
                           if (datetime.now() - datetime.fromisoformat(e['timestamp'])).seconds < 300]

            if len(recent_errors) >= 3:
                alerts.append("⚠️ 최근 5분간 에러가 3회 이상 발생했습니다.")

        return alerts

    def reset_metrics(self):
        """메트릭 초기화"""
        with self.lock:
            self.metrics = {
                'query_count': 0,
                'success_count': 0,
                'error_count': 0,
                'total_response_time': 0.0,
                'company_usage': defaultdict(int),
                'popular_queries': defaultdict(int),
                'recent_errors': [],
                'hourly_stats': defaultdict(lambda: {'queries': 0, 'errors': 0, 'response_time': 0.0})
            }
            self.start_time = datetime.now()
            print("✅ 모니터링 메트릭이 초기화되었습니다.")


class GPTQualityEvaluator:
    """GPT 기반 품질 평가 시스템"""

    def __init__(self, llm):
        self.llm = llm
        self.test_queries = [
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

    def evaluate_quality(self, companies: List[str], rag_chain_func) -> str:
        """GPT 기반 품질 평가 실행"""
        try:
            results_text = "# 🔬 GPT 기반 시스템 평가 결과\n\n"
            results_text += f"**평가 대상**: {', '.join(companies)}\n\n"

            for company in companies:
                try:
                    # 랜덤 쿼리 선택
                    selected_query = random.choice(self.test_queries)

                    results_text += f"## 📋 {company}\n"
                    results_text += f"**테스트 질문**: {selected_query}\n\n"

                    # RAG 시스템으로 답변 생성
                    result = rag_chain_func(selected_query, [company])
                    
                    if isinstance(result, dict):
                        answer = result.get('answer', 'No answer')
                        sources = result.get('sources', [])
                    else:
                        answer = str(result)
                        sources = []

                    results_text += f"**RAG 시스템 답변**:\n{answer}\n\n"

                    if sources:
                        source_info = []
                        for src in sources[:2]:  # 상위 2개만 표시
                            if isinstance(src, dict):
                                page_str = f"p.{src.get('page', 'n/a')}" if src.get('page') not in (None, "n/a") else "p.n/a"
                                source_info.append(f"{src.get('doc_id', 'unknown')} / {page_str}")
                            else:
                                source_info.append(str(src))
                        results_text += f"**출처**: {', '.join(source_info)}\n\n"

                    # GPT를 통한 품질 평가
                    evaluation_prompt = f"""
다음은 펫보험 관련 질문과 RAG 시스템의 답변입니다.

질문: {selected_query}
답변: {answer}

이 답변을 다음 기준으로 평가해주세요:
1. 정확성: 질문에 정확히 답변했는가?
2. 완성도: 답변이 충분히 상세한가?
3. 유용성: 사용자에게 실질적으로 도움이 되는가?
4. 명확성: 답변이 이해하기 쉬운가?

평가 결과를 자연스러운 문장으로 작성해주세요. 점수보다는 구체적인 피드백을 중심으로 해주세요.
"""

                    # GPT 평가 요청
                    evaluation_response = self.llm.invoke(evaluation_prompt)
                    if hasattr(evaluation_response, 'content'):
                        gpt_evaluation = evaluation_response.content
                    else:
                        gpt_evaluation = str(evaluation_response)

                    results_text += f"**🤖 GPT 품질 평가**:\n{gpt_evaluation}\n\n"
                    results_text += "---\n\n"

                except Exception as e:
                    results_text += f"**❌ {company} 평가 실패**: {str(e)}\n\n"
                    continue

            results_text += "### 📈 종합 평가 완료\n"
            results_text += "각 회사별로 랜덤 선택된 질문에 대한 GPT 기반 품질 평가가 완료되었습니다.\n"
            results_text += "평가 결과를 참고하여 시스템 개선에 활용하세요.\n"

            return results_text

        except Exception as e:
            return f"❌ 평가 중 오류가 발생했습니다: {str(e)}"


def create_evaluation_system(db_path: str = None):
    """평가 시스템 인스턴스 생성"""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "user_feedback.db"
    
    user_feedback_evaluator = UserFeedbackEvaluator(str(db_path))
    ab_test_evaluator = ABTestEvaluator()
    real_time_monitor = RealTimeMonitor()
    
    return user_feedback_evaluator, ab_test_evaluator, real_time_monitor


if __name__ == "__main__":
    print("✅ 평가 시스템 모듈 로드 완료!")
    print("  - UserFeedbackEvaluator: 피드백 수집 및 분석")
    print("  - ABTestEvaluator: A/B 테스트 관리")
    print("  - RealTimeMonitor: 실시간 성능 모니터링")
    print("  - GPTQualityEvaluator: GPT 기반 품질 평가")