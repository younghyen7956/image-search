import os
from pathlib import Path
from io import BytesIO
from PIL import Image
import gradio as gr
import base64

BASE_DIR  = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"    # 실제 이미지들이 저장된 디렉토리 # 실제 서비스에서는 s3 버킷 주

from image_search.service.image_search_service_impl import ImageSearchServiceImpl
service = ImageSearchServiceImpl.getInstance()

async def image_search_gradio(input_image):
    # 3-1) PIL.Image → BytesIO
    buf = BytesIO() # 메모리에 올려서 이미지 데이터를 사용 SpooledTemporaryFile(max_size=1024*1024) 이걸로 사용하면 디스크에 스풀해서 사용 가능
    input_image.save(buf, format="PNG")
    buf.seek(0) # 파일을 저장하면 커서가 파일 뒷편으로 가게 되서 커서를 제일 앞쪽으로 이동하는 코드 이 코드가 없다면 파일이 없다고 인식하게 됨

    class DummyUploadFile:
        filename = "query.png"
        def __init__(self, buffer):
            self.file = buffer

        async def read(self):
            # UploadFile.read()처럼 동작하도록
            return self.file.read()

        async def seek(self, pos: int):
            return self.file.seek(pos)

    upload_file = DummyUploadFile(buf)

    # 3-3) 서비스 호출 (벡터 변환 → Milvus 검색)
    results = await service.imageSearch(upload_file, top_k=3)

    # 3-4) 검색 결과 이미지 로드
    data_uris = []
    for r in results:
        img_path = IMAGE_DIR / r["filename"]
        buf = BytesIO()
        if img_path.exists():
            Image.open(img_path).save(buf, "PNG")
        else:
            Image.new("RGB", (224, 224), (255, 0, 0)).save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uris.append(f"data:image/png;base64,{b64}")

    return [{"image": uri} for uri in data_uris]

with gr.Blocks() as demo:
    inp = gr.Image(type="pil", label="검색할 이미지 업로드")
    out = gr.Gallery(label="검색 결과", columns=5, height="auto")
    gr.Button("검색 시작").click(image_search_gradio, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3333)
