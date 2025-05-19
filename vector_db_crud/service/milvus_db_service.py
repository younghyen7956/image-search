from typing import List, Optional
from abc import ABC, abstractmethod
from pymilvus import MilvusException
from pymilvus.grpc_gen.milvus_pb2_grpc import MilvusService


class MilvusDBService(ABC):

    @abstractmethod
    def create_collection(
        self,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        fields: Optional[List] = None,
        overwrite: bool = False
    ) -> None:
        pass

    @abstractmethod
    def drop_collection(self, name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def upsert(self, data: List[dict],collection_name: str) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 3,
        metric: str = "COSINE",
        nprobe: int = 10,
        output_fields: Optional[List[str]] = None,
        collection_name: str = None
    ) -> List[dict]:
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[int],collection_name: str) -> None:
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        pass

    @abstractmethod
    def describe_collection(self, name: Optional[str] = None) -> dict:
        pass

    @abstractmethod
    def query_entities(self, collection_name: str,top_k: int = 5, ids: int =1) -> List[dict]:
        pass