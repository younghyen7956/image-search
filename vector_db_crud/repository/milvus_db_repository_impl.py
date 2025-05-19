import os
from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient,
    MilvusException
)
import numpy as np

from vector_db_crud.repository.milvus_db_repository import MilvusDBRepository

class MilvusDBRepositoryImpl(MilvusDBRepository):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            # .env 로딩은 인스턴스 최초 생성 시 1회만
            load_dotenv()
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self,
                 host: str = None, port: str = None,
                 collection_name: str = None,
                 dim: int = None):
        # 1) 환경변수 또는 인자로 연결 정보 설정
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "images")
        self.dim = dim  # 임베딩 차원 (새로 생성 시 필요)

        # 2) Milvus 연결
        connections.connect(alias="default", host=self.host, port=self.port)

    def create_collection(
        self,
        name: str = None,
        dim: int = None,
        fields: list = None,
        overwrite: bool = False
    ):
        """스키마 정의 후 컬렉션 생성 및 인덱스 자동 생성."""
        coll_name = name or self.collection_name
        coll_dim  = dim or self.dim
        client = MilvusClient(host=self.host, port=self.port)

        if coll_dim is None and not fields:
            raise ValueError("'dim' 또는 'fields' 중 하나는 반드시 제공해야 합니다.")

        # 기존 컬렉션 존재 체크
        if utility.has_collection(coll_name):
            if overwrite:
                utility.drop_collection(coll_name)
            else:
                print(f"[create_collection] '{coll_name}' 이미 존재합니다.")
                return

        # 기본 필드 정의
        if fields is None:
            fields = [
                FieldSchema(name="id",dtype=DataType.INT64,is_primary=True,auto_id=False),
                FieldSchema(name="embedding",dtype=DataType.FLOAT_VECTOR, dim=coll_dim),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512)
            ]
        else:
            if not all(isinstance(f, FieldSchema) for f in fields):
                raise ValueError("fields는 FieldSchema 객체들의 리스트여야 합니다.")

        # 1) 빈 스키마 생성
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # 2) fields 리스트를 순회하며 add_field 호출
        for f in fields:
            params = {}
            # FieldSchema 의 속성들을 꺼내서 적절하게 매핑
            if f.dtype == DataType.VARCHAR:
                params["max_length"] = f.max_length
            if f.dtype == DataType.FLOAT_VECTOR:
                params["dim"] = f.dim

            schema.add_field(
                field_name = f.name,
                datatype   = f.dtype,
                is_primary = getattr(f, "is_primary", False),
                auto_id    = getattr(f, "auto_id", False),
                **params
            )

        print(f"[create_collection] 스키마 필드 추가 완료: {[f.name for f in fields]}")

        # 3) 인덱스 파라미터 준비
        idx_params = client.prepare_index_params()

        # 벡터 필드에 IVF_FLAT + COSINE 인덱스
        idx_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
            index_name="vector_index"
        )

        # 4) 컬렉션 생성 (스키마 + 인덱스)
        client.create_collection(
            collection_name=coll_name,
            schema=schema,
            index_params=idx_params
        )
        print(f"[create_collection] '{coll_name}' 생성됨. 인덱스 설정: {idx_params}")
        print(f"[create_collection] 인덱스 빌드 완료.")

    def drop_collection(self, name: str = None):
        """컬렉션 완전 삭제"""
        coll_name = name or self.collection_name
        client = MilvusClient(host=self.host, port=self.port)
        client.drop_collection(collection_name=coll_name)
        print(f"[drop_collection] Dropped '{coll_name}'")

    def upsert(self, data: list[dict], collection_name: str = None):
        client = MilvusClient(host=self.host, port=self.port)

        try:
            # 실제 upsert 호출
            result = client.upsert(
                collection_name=collection_name or self.collection_name,
                data=data
            )
        except MilvusException as me:
            print(f"[upsert][ERROR] MilvusException: code={me.code}, reason={me.message}")
            return None
        except Exception as e:
            print(f"[upsert][ERROR] Unexpected error: {e}")
            return None

        # 성공 시 로깅
        count = len(data)
        print(f"[upsert] Upserted {count} entities into '{collection_name or self.collection_name}'.")
        return result

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
        """
        벡터 검색 후, 추가 필드를 포함해 결과를 포맷하여 반환합니다.
        """
        client = MilvusClient(host=self.host, port=self.port)
        # MilvusClient.search 에 넘길 파라미터
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": nprobe}
        }

        try:
            res = client.search(
                collection_name=collection_name or self.collection_name,
                anns_field=anns_field,
                data=query_vectors,
                limit=top_k,
                search_params=search_params,           # 수정된 부분
                output_fields=output_fields or ["filename", "embedding"],
            )
        except MilvusException as me:
            print(f"[imageSearch][ERROR] MilvusException: code={me.code}, reason={me.message}")
            return []
        except Exception as e:
            print(f"[imageSearch][ERROR] Unexpected error: {e}")
            return []

        # 결과 포맷
        fields = output_fields or ["filename", "embedding"]
        formatted = []
        for q_idx, hits in enumerate(res):
            for hit in hits:
                record = {
                    "query_index": q_idx,
                    "id": hit.id,
                    "score": hit.distance
                }
                # 추가 필드 삽입
                for fld in fields:
                    record[fld] = hit.entity.get(fld)
                formatted.append(record)

        return formatted

    def delete_by_id(self, ids: list[int],collection_name: str) -> None:
        """Primary key(img_id)로 데이터 삭제"""
        client = MilvusClient(host=self.host, port=self.port)
        client.delete(
            collection_name=collection_name or self.collection_name,
            ids=ids
        )
        print(f"[delete_by_id] Deleted ids={ids} from '{collection_name or self.collection_name}'")

    def list_collections(self):
        client = MilvusClient(host=self.host, port=self.port)
        result = client.list_collections()
        return result

    def describe(self, collection_name: str = None):
        # 1) 사용할 컬렉션 이름 결정
        coll_name = collection_name or self.collection_name
        if coll_name is None:
            raise ValueError("describe 호출 시 collection_name을 지정하거나, "
                             "먼저 create_collection으로 기본 컬렉션을 설정해야 합니다.")

        client = MilvusClient(host=self.host, port=self.port)
        # 3) 스키마 정보 조회
        res = client.describe_collection(collection_name=coll_name)
        return res

    def query(self, collection_name: str, top_k: int,ids: int):
        col = Collection(name=collection_name or self.collection_name)
        raw = col.query(
            expr=f'id>={ids}',
            output_fields=['id','filename','embedding'],
            limit=top_k
        )

        # numpy → native 변환
        formatted = []
        for row in raw:
            # row 는 {'id': numpy.int64, 'filename': str, 'embedding': list[numpy.float32] }
            # 1) id: int
            pk = int(row['id'])
            # 2) filename: str 그대로
            fn = row['filename']
            # 3) embedding: list → list[float]
            emb = row['embedding']
            # 만약 numpy.ndarray 로 오는 경우
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = [float(x) for x in emb]

            formatted.append({
                'id': pk,
                'filename': fn,
                'embedding': emb_list
            })

        return formatted
