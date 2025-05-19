import gradio as gr
import requests
import pandas as pd
import json

API_BASE = "http://localhost:3333"

# 컬렉션 리스트 가져오기
def list_collections():
    res = requests.post(f"{API_BASE}/collections_list")
    return "\n".join(res.json().get("collections", []))

# 컬렉션 생성
def create_collection(name, dim):
    params = {"name": name, "dim": int(dim), "overwrite": True}
    r = requests.post(f"{API_BASE}/create", params=params)
    return r.json().get("message", r.text)

# 엔티티 업서트 (JSON 또는 CSV 둘 다 가능)
def upsert_entities(collection_name, json_text, csv_file):
    entities = []

    # 1) JSON 입력 우선 처리
    js = (json_text or "").strip()
    if js:
        try:
            entities = json.loads(js)
            if not isinstance(entities, list):
                return "JSON 입력은 리스트 형태여야 합니다."
        except Exception as e:
            return f"JSON 파싱 오류: {e}"
    # 2) 아니면 CSV 처리
    elif csv_file:
        try:
            df = pd.read_csv(csv_file.name)
        except Exception as e:
            return f"CSV 로드 오류: {e}"

        # CSV에는 id, filename, embedding 컬럼이 있어야 한다고 가정
        if not {"id","filename","embedding"}.issubset(df.columns):
            return "CSV에 id, filename, embedding 컬럼이 모두 있어야 합니다."

        for _, row in df.iterrows():
            try:
                emb = [float(x) for x in str(row["embedding"]).split(",")]
            except Exception as e:
                return f"Embedding 파싱 오류: {e}"
            entities.append({
                "id":        int(row["id"]),
                "filename":  str(row["filename"]),
                "embedding": emb
            })
    else:
        return "JSON 텍스트 또는 CSV 파일 중 하나를 입력하세요."

    # 3) FastAPI /upsert 호출
    try:
        resp = requests.post(
            f"{API_BASE}/upsert",
            params={"collection_name": collection_name},
            json=entities,
            timeout=10
        )
    except Exception as e:
        return f"Upsert 요청 실패: {e}"

    try:
        return resp.json().get("message", resp.text)
    except:
        return resp.text

# 벡터 검색
def search_vectors(collection_name, top_k, nprobe, embedding_str):
    try:
        query_vec = [float(x) for x in embedding_str.strip().split(",")]
    except:
        return "임베딩을 콤마로 구분된 숫자 리스트로 입력하세요."
    payload = {
        "query_vectors": [query_vec],
        "top_k": int(top_k),
        "nprobe": int(nprobe)
    }
    r = requests.post(
        f"{API_BASE}/search",
        params={"collection_name": collection_name},
        json=payload
    )
    return r.json().get("results", r.text)
def query_collection(collection_name: str, top_k: int):
    """
    FastAPI의 /query_collection 엔드포인트를 호출해서
    지정한 컬렉션에서 top_k 개수만큼 엔티티를 조회합니다.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/query_collection",
            params={"collection_name": collection_name, "top_k": top_k},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("entities", [])
    except Exception as e:
        return f"Query 요청 실패: {e}"

def describe_collection_gr(collection_name: str):
    """
    FastAPI /describe 엔드포인트 호출 → 컬렉션 스키마(JSON) 반환
    """
    try:
        resp = requests.post(
            f"{API_BASE}/describe",
            params={"collection_name": collection_name},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("schema", {})
    except Exception as e:
        return {"error": str(e)}


def drop_collection_gr(collection_name: str) -> str:
    """
    FastAPI DELETE /drop_collection?collection_name=<name> 호출
    """
    try:
        resp = requests.delete(
            f"{API_BASE}/drop_collection",
            params={"collection_name": collection_name},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("message", resp.text)
    except Exception as e:
        return f"Drop Collection 요청 실패: {e}"

def delete_entities_gr(collection_name: str, ids_str: str) -> str:
    """
    - collection_name: 삭제할 컬렉션 이름
    - ids_str: "1,2,3" 처럼 콤마로 구분된 ID 문자열
    """
    # 1) ids_str → List[int]
    try:
        ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
        if not ids:
            return "삭제할 ID를 콤마로 구분해 하나 이상 입력하세요."
    except Exception as e:
        return f"IDs 파싱 오류: {e}"

    # 2) DELETE 요청
    try:
        resp = requests.delete(
            f"{API_BASE}/delete",
            params={"collection_name": collection_name},
            json={"ids": ids},
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Delete 요청 실패: {e}"

    # 3) 결과 메시지 반환
    try:
        return resp.json().get("message", resp.text)
    except:
        return resp.text

with gr.Blocks() as demo:
    gr.Markdown("## Vector DB 관리 UI (Gradio)")

    # Collections 탭
    with gr.Tab("Collections"):
        with gr.Row():
            col_list = gr.Textbox(label="컬렉션 목록", interactive=False)
            btn_list = gr.Button("리스트 갱신")
        btn_list.click(fn=list_collections, outputs=col_list)

        with gr.Row():
            name_in    = gr.Textbox(label="생성할 Collection 이름")
            dim_in     = gr.Number(label="벡터 차원", value=128)
            btn_create = gr.Button("생성/재생성")
        btn_create.click(fn=create_collection, inputs=[name_in, dim_in], outputs=col_list)

    # Upsert 탭 (JSON or CSV)
    with gr.Tab("Upsert"):
        with gr.Row():
            col_name_up = gr.Textbox(label="Collection 이름")
        with gr.Row():
            json_in     = gr.Textbox(
                label="직접 JSON 입력",
                placeholder='[{"id":1,"filename":"a.png","embedding":[0.1,0.2,...]}]',
                lines=4
            )
        with gr.Row():
            csv_up      = gr.File(label="또는 CSV 파일 업로드 (.csv)")
        with gr.Row():
            btn_upsert  = gr.Button("Upsert 실행")
            msg_upsert  = gr.Textbox(label="결과 메시지", interactive=False)
        btn_upsert.click(
            fn=upsert_entities,
            inputs=[col_name_up, json_in, csv_up],
            outputs=msg_upsert
        )

    # Search 탭
    with gr.Tab("Search"):
        with gr.Row():
            col_name_s = gr.Textbox(label="Collection 이름")
        with gr.Row():
            topk_s     = gr.Slider(1, 20, value=5, label="Top K")
            nprobe_s   = gr.Slider(1, 64, value=10, label="Nprobe")
        with gr.Row():
            emb_str    = gr.Textbox(label="Query 임베딩 (콤마 구분)")
        with gr.Row():
            btn_search = gr.Button("검색")
            out_search = gr.JSON(label="검색 결과")
        btn_search.click(
            fn=search_vectors,
            inputs=[col_name_s, topk_s, nprobe_s, emb_str],
            outputs=out_search
        )

    with gr.Tab("Query"):
        with gr.Row():
            col_name_q = gr.Textbox(label="Collection 이름", value="test")
            topk_q = gr.Slider(1, 20, value=5, label="Top K 조회 개수")
        with gr.Row():
            btn_query = gr.Button("Query 실행")
            out_query = gr.JSON(label="조회 결과")
        btn_query.click(
            fn=query_collection,
            inputs=[col_name_q, topk_q],
            outputs=out_query
        )

    with gr.Tab("Describe"):
        col_name_d = gr.Textbox(label="Collection 이름", value="test")
        btn_describe = gr.Button("스키마 조회")
        out_schema = gr.JSON(label="스키마 정보")
        btn_describe.click(
            fn=describe_collection_gr,
            inputs=[col_name_d],
            outputs=out_schema
        )

    with gr.Tab("Drop Collection"):
        col_name_drop = gr.Textbox(label="삭제할 Collection 이름", value="test")
        btn_drop = gr.Button("삭제 실행")
        out_drop = gr.Textbox(label="결과 메시지", interactive=False)

        btn_drop.click(
            fn=drop_collection_gr,
            inputs=[col_name_drop],
            outputs=out_drop
        )

    with gr.Tab("Delete Entities"):
        col_name_del = gr.Textbox(label="Collection 이름", value="test")
        ids_in = gr.Textbox(label="삭제할 ID들 (콤마 구분)", placeholder="1,2,3")
        btn_del = gr.Button("엔티티 삭제 실행")
        out_del = gr.Textbox(label="결과 메시지", interactive=False)

        btn_del.click(
            fn=delete_entities_gr,
            inputs=[col_name_del, ids_in],
            outputs=out_del
        )

if __name__ == "__main__":
    demo.launch(share=False,server_name="0.0.0.0", server_port=8002)
