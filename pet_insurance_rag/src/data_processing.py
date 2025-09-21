# =========================================
# 📊 데이터 처리 및 벡터 데이터베이스 모듈
# =========================================

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

# LangChain 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

# 설정 임포트
try:
    from config.settings import *
except ImportError:
    # 기본 설정값
    VECTOR_BACKEND = "faiss"
    OPENAI_EMB_MODEL = "text-embedding-ada-002"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class PetInsuranceGuard:
    """펫보험 약관 전처리 및 정제 클래스"""
    
    def __init__(self):
        # 불용어 및 필터링 규칙
        self.stop_words = {
            '그', '이', '저', '것', '들', '수', '있', '없', '하', '되', '된', '될', '함', 
            '및', '등', '의', '를', '에', '로', '과', '와', '은', '는', '이', '가',
            '보험', '가입', '계약', '약관', '조항', '규정', '내용', '사항', '경우',
            '때문', '따라', '관련', '대한', '위한', '통해', '이용', '제공', '처리'
        }
        
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        
        return text
    
    def filter_relevant_content(self, text: str) -> bool:
        """펫보험 관련 내용만 필터링"""
        if not text or len(text) < 10:  # 너무 짧은 텍스트 제외
            return False
            
        # 기본적으로 모든 보험 관련 내용을 포함
        insurance_keywords = [
            '보험', '보장', '약관', '계약', '가입', '청구', '지급', '보상',
            '치료', '진료', '의료', '병원', '수술', '입원', '통원',
            '질병', '상해', '사고', '부상', '예방', '건강',
            '반려동물', '펫', '애완동물', '강아지', '고양이', '동물'
        ]
        
        text_lower = text.lower()
        
        # 보험 관련 키워드가 하나라도 있으면 포함
        if any(keyword in text_lower for keyword in insurance_keywords):
            return True
        
        # 숫자나 금액이 포함된 경우도 포함 (보험금 관련)
        if any(char.isdigit() for char in text) and ('원' in text or '만' in text or '%' in text):
            return True
        
        return False
    
    def extract_structured_info(self, text: str) -> Dict[str, Any]:
        """구조화된 정보 추출"""
        info = {
            'has_coverage_info': False,
            'has_exclusion_info': False,
            'has_procedure_info': False,
            'coverage_percentage': None,
            'max_amount': None
        }
        
        text_lower = text.lower()
        
        coverage_keywords = ['보장', '지급', '보상', '급여']
        if any(keyword in text_lower for keyword in coverage_keywords):
            info['has_coverage_info'] = True
        
        exclusion_keywords = ['면책', '제외', '보장하지', '지급하지']
        if any(keyword in text_lower for keyword in exclusion_keywords):
            info['has_exclusion_info'] = True
        
        procedure_keywords = ['신청', '접수', '제출', '청구']
        if any(keyword in text_lower for keyword in procedure_keywords):
            info['has_procedure_info'] = True
        
        return info

class LangChainEmbeddingsManager:
    """임베딩 및 캐시 관리 클래스"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or OPENAI_EMB_MODEL
        self.embeddings = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """캐시 기반 임베딩 설정"""
        # API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # 기본 임베딩 모델
        base_embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=api_key
        )
        
        # 캐시 저장소 설정
        cache_dir = "./embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        store = LocalFileStore(cache_dir)
        
        # 캐시 기반 임베딩
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            base_embeddings, 
            store,
            namespace=f"openai_{self.model_name}"
        )
        
        print(f"✅ 캐시 기반 임베딩 설정 완료: {self.model_name}")
    
    def get_embeddings(self):
        """임베딩 인스턴스 반환"""
        return self.embeddings

class CompanyVectorStoreManager:
    """회사별 벡터 스토어 관리 클래스"""
    
    def __init__(self, embeddings_manager: LangChainEmbeddingsManager):
        self.embeddings = embeddings_manager.get_embeddings()
        self.guard = PetInsuranceGuard()
        self.company_vector_stores = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        # CSV 파일 매핑
        from config.settings import CSV_FILES, DATA_DIR
        self.csv_files = {}
        for company, filename in CSV_FILES.items():
            file_path = DATA_DIR / filename
            self.csv_files[company] = str(file_path)

        
    def load_company_data(self, company: str, csv_file: str) -> List[Document]:
        """회사별 CSV 데이터 로드 및 전처리"""
        print(f"📊 {company} 데이터 로딩 중...")
        
        if not os.path.exists(csv_file):
            print(f"⚠️ 파일을 찾을 수 없습니다: {csv_file}")
            return []
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='cp949')
            except Exception as e:
                print(f"❌ {csv_file} 읽기 실패: {e}")
                return []
        
        documents = []
        processed_count = 0
        
        for idx, row in df.iterrows():
            # CSV 구조에 맞는 텍스트 추출
            text = None
            
            # 실제 CSV 컬럼에 맞게 수정
            if 'chunk_text' in df.columns and pd.notna(row['chunk_text']):
                text = str(row['chunk_text'])
            elif 'text' in df.columns and pd.notna(row['text']):
                text = str(row['text'])
            elif 'content' in df.columns and pd.notna(row['content']):
                text = str(row['content'])
            else:
                # 다른 텍스트 컬럼 시도
                text_columns = ['claim_sentence', 'preview', 'data', 'document']
                for col in text_columns:
                    if col in df.columns and pd.notna(row[col]):
                        text = str(row[col])
                        break
            
            if not text or len(text.strip()) < 10:
                continue
            
            # 텍스트 정제
            cleaned_text = self.guard.clean_text(text)
            
            # 관련성 필터링
            if not self.guard.filter_relevant_content(cleaned_text):
                continue
            
            # 구조화된 정보 추출
            structured_info = self.guard.extract_structured_info(cleaned_text)
            
            # 메타데이터 구성
            metadata = {
                'company': company,
                'source': os.path.basename(csv_file),
                'chunk_id': row.get('chunk_id', idx),
                'row_index': idx,
                **structured_info
            }
            
            # 추가 메타데이터
            for col in ['page', 'section_name', 'subsection_name', 'subject', 'procedure']:
                if col in df.columns and pd.notna(row[col]):
                    metadata[col] = row[col]
            
            # Document 생성
            doc = Document(
                page_content=cleaned_text,
                metadata=metadata
            )
            
            documents.append(doc)
            processed_count += 1
        
        print(f"✅ {company}: {processed_count}개 문서 처리 완료")
        return documents
    
    def create_company_vector_store(self, company: str, documents: List[Document]) -> FAISS:
        """회사별 벡터 스토어 생성"""
        print(f"🔍 {company} 벡터 스토어 생성 중...")
        
        if not documents:
            raise ValueError(f"{company}에 대한 유효한 문서가 없습니다.")
        
        # 텍스트 분할
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        print(f"📑 {company}: {len(all_chunks)}개 청크 생성")
        
        # 배치 처리로 벡터 스토어 생성
        batch_size = 50
        batches = [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]
        
        vector_store = None
        
        try:
            first_batch = batches[0]
            vector_store = FAISS.from_documents(first_batch, self.embeddings)
            print(f"   ✅ 배치 1/{len(batches)} 완료 ({len(first_batch)}개 문서)")
        except Exception as e:
            print(f"   ❌ 첫 번째 배치 실패: {e}")
            raise
        
        # 나머지 배치들 추가
        for i, batch in enumerate(batches[1:], 2):
            try:
                batch_vector_store = FAISS.from_documents(batch, self.embeddings)
                vector_store.merge_from(batch_vector_store)
                print(f"   ✅ 배치 {i}/{len(batches)} 완료 ({len(batch)}개 문서)")
            except Exception as e:
                print(f"   ⚠️ 배치 {i} 실패, 건너뜀: {e}")
                continue
        
        print(f"🎉 {company} 벡터 스토어 생성 완료!")
        return vector_store
    
    def load_all_companies(self) -> Dict[str, FAISS]:
        """모든 회사 데이터 로드 및 벡터 스토어 생성"""
        print("🚀 전체 회사 데이터 로딩 시작...")
        
        for company, csv_file in self.csv_files.items():
            if not os.path.exists(csv_file):
                print(f"⚠️ {csv_file} 파일을 찾을 수 없습니다. {company} 건너뜀.")
                continue
            
            try:
                # 데이터 로드
                documents = self.load_company_data(company, csv_file)
                
                if documents:
                    # 벡터 스토어 생성
                    vector_store = self.create_company_vector_store(company, documents)
                    self.company_vector_stores[company] = vector_store
                else:
                    print(f"⚠️ {company}: 유효한 문서가 없습니다.")
            
            except Exception as e:
                print(f"❌ {company} 처리 실패: {e}")
                continue
        
        total_companies = len(self.company_vector_stores)
        total_docs = sum(len(vs.docstore._dict) for vs in self.company_vector_stores.values())
        
        print(f"🎉 전체 로딩 완료!")
        print(f"📊 처리된 회사: {total_companies}개")
        print(f"📄 총 문서 수: {total_docs}개")
        
        return self.company_vector_stores

class CompanyFilteredRetriever:
    """회사별 필터링된 검색 클래스"""
    
    def __init__(self, company_vector_stores: Dict[str, FAISS]):
        self.company_vector_stores = company_vector_stores
        self.bm25_retrievers = {}
        self._setup_bm25_retrievers()
    
    def _setup_bm25_retrievers(self):
        """회사별 BM25 검색기 설정"""
        print("🔍 회사별 BM25 검색기 설정 중...")
        
        for company, vector_store in self.company_vector_stores.items():
            try:
                docs = list(vector_store.docstore._dict.values())
                
                if docs:
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 5
                    self.bm25_retrievers[company] = bm25_retriever
                    print(f"   ✅ {company} BM25 검색기 설정 완료")
                else:
                    print(f"   ⚠️ {company}: 문서가 없어 BM25 검색기 생성 실패")
                    
            except Exception as e:
                print(f"   ❌ {company} BM25 검색기 생성 실패: {e}")
    
    def get_retriever(self, company: str = None, k: int = 5):
        """회사별 검색기 반환"""
        if company and company in self.company_vector_stores:
            vector_store = self.company_vector_stores[company]
            try:
                return vector_store.as_retriever(search_kwargs={"k": k})
            except Exception as e:
                print(f"⚠️ {company} 검색기 생성 실패: {e}")
                return vector_store.as_retriever(search_kwargs={"k": k})
        else:
            if self.company_vector_stores:
                first_company = list(self.company_vector_stores.keys())[0]
                return self.company_vector_stores[first_company].as_retriever(search_kwargs={"k": k})
            else:
                raise ValueError("사용 가능한 벡터 스토어가 없습니다.")

def format_docs(docs: List[Document]) -> str:
    """문서들을 안전하게 포맷팅"""
    if not docs:
        return ""
    
    formatted_docs = []
    for doc in docs:
        if doc is None:
            continue
            
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        if hasattr(doc, 'metadata') and doc.metadata:
            meta = doc.metadata
            company = meta.get('company', 'Unknown')
            source = meta.get('source', 'Unknown')
            formatted_docs.append(f"[{company}] {content}")
        else:
            formatted_docs.append(content)
    
    return "\n\n".join(formatted_docs)

def initialize_data_processing():
    """데이터 처리 모듈 초기화"""
    print("🔧 데이터 처리 모듈 초기화 중...")
    
    # API 키 확인 (환경변수 우선)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # 백업: openaikey.txt 파일에서 읽기 시도
        try:
            with open('openaikey.txt', 'r') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ OpenAI API 키를 백업 파일(openaikey.txt)에서 로드했습니다.")
        except FileNotFoundError:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. 다음 중 하나를 선택하세요:\n"
                "1. .env 파일에 OPENAI_API_KEY=your-key 추가\n"
                "2. 환경변수 OPENAI_API_KEY 설정\n"
                "3. openaikey.txt 파일 생성"
            )
    else:
        print("✅ OpenAI API 키를 환경변수에서 로드했습니다.")
    
    # 임베딩 매니저 초기화
    embeddings_manager = LangChainEmbeddingsManager()
    
    # 벡터 스토어 매니저 초기화
    vector_manager = CompanyVectorStoreManager(embeddings_manager)
    
    # 모든 회사 데이터 로드
    company_vector_stores = vector_manager.load_all_companies()
    
    # 검색기 설정
    filtered_retriever = CompanyFilteredRetriever(company_vector_stores)
    
    # 회사별 개별 검색기 생성
    filtered_retrievers = {}
    for company in company_vector_stores.keys():
        filtered_retrievers[company] = filtered_retriever.get_retriever(company)
    
    print("✅ 데이터 처리 모듈 초기화 완료!")
    
    return {
        'embeddings': embeddings_manager.get_embeddings(),
        'company_vector_stores': company_vector_stores,
        'filtered_retrievers': filtered_retrievers,
        'embeddings_manager': embeddings_manager,
        'vector_manager': vector_manager,
        'filtered_retriever': filtered_retriever
    }

if __name__ == "__main__":
    # 테스트 실행
    result = initialize_data_processing()
    print(f"📊 처리된 회사 수: {len(result['company_vector_stores'])}")
    print(f"🔍 검색기 수: {len(result['filtered_retrievers'])}")