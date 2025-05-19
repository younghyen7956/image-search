from typing import List, Optional

from pymilvus import MilvusException

from vector_db_crud.repository.milvus_db_repository_impl import MilvusDBRepositoryImpl
from vector_db_crud.service.milvus_db_service import MilvusDBService


class MilvusDBServiceImpl(MilvusDBService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__MilvusDBRepositoryImpl = MilvusDBRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def create_collection(
        self,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        fields: Optional[List] = None,
        overwrite: bool = False
    ) -> None:
        """
        컬렉션 생성 및 인덱스 자동 생성.
        """
        try:
            self.__MilvusDBRepositoryImpl.create_collection(
                name=name,
                dim=dim,
                fields=fields,
                overwrite=overwrite
            )
        except MilvusException as e:
            raise RuntimeError(f"컬렉션 생성 실패: {e}")

    def drop_collection(self, name: Optional[str] = None) -> None:
        """컬렉션 삭제"""
        try:
            self.__MilvusDBRepositoryImpl.drop_collection(name=name)
        except MilvusException as e:
            raise RuntimeError(f"컬렉션 삭제 실패: {e}")

    def upsert(self, data: List[dict],collection_name: str) -> None:
        """데이터 삽입 또는 갱신"""
        result = self.__MilvusDBRepositoryImpl.upsert(data=data,collection_name=collection_name)
        if result is None:
            raise RuntimeError("업서트 중 오류가 발생했습니다.")

    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 3,
        metric: str = "COSINE",
        nprobe: int = 10,
        output_fields: Optional[List[str]] = None,
        collection_name: str = None
    ) -> List[dict]:
        """벡터 검색 및 결과 반환"""
        try:
            return self.__MilvusDBRepositoryImpl.imageSearch(
                query_vectors=query_vectors,
                top_k=top_k,
                metric_type=metric,
                nprobe=nprobe,
                output_fields=output_fields,
                collection_name=collection_name
            )
        except MilvusException as e:
            raise RuntimeError(f"검색 실패: {e}")

    def delete_by_ids(self, ids: List[int],collection_name: str) -> None:
        """Primary key 리스트로 삭제"""
        try:
            self.__MilvusDBRepositoryImpl.delete_by_id(ids=ids,collection_name=collection_name)
        except MilvusException as e:
            raise RuntimeError(f"삭제 실패: {e}")

    def list_collections(self) -> List[str]:
        """등록된 모든 컬렉션 이름 반환"""
        return self.__MilvusDBRepositoryImpl.list_collections()

    def describe_collection(self, name: Optional[str] = None) -> dict:
        """컬렉션 스키마 정보 반환"""
        return self.__MilvusDBRepositoryImpl.describe(collection_name=name)

    def query_entities(self, collection_name: str,top_k: int = 5, ids: int=1) -> List[dict]:
        """컬렉션의 처음 top_k 엔티티를 조회"""
        return self.__MilvusDBRepositoryImpl.query(collection_name= collection_name,top_k=top_k,ids=ids)
