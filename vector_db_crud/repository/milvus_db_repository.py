from abc import ABC, abstractmethod


class MilvusDBRepository(ABC):

    @abstractmethod
    def create_collection(
        self,
        name: str = None,
        dim: int = None,
        fields: list = None,
        overwrite: bool = False
    ):
        pass

    @abstractmethod
    def drop_collection(self, name: str = None):
        pass

    @abstractmethod
    def upsert(self, data: list[dict], collection_name: str = None):
        pass

    @abstractmethod
    def imageSearch(
        self,
        query_vectors: list[list[float]],
        anns_field: str = "embedding",
        output_fields: list[str] = None,
        top_k: int = 3,
        metric_type: str = "COSINE",
        nprobe: int = 10,
        collection_name: str = None,
    ) -> list[dict]:
        pass

    def delete_by_id(self, ids: list[int],collection_name: str) -> None:
        pass

    def list_collections(self):
        pass

    def describe(self, collection_name: str = None):
        pass

    def query(self, collection_name: str, top_k: int,ids: int):
        pass