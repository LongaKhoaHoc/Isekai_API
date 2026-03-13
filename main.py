
from fastapi import FastAPI
from pydantic import BaseModel
import os
from groq import Groq

app = FastAPI()

# Lên Server mình sẽ giấu API Key của sếp vào đây cho an toàn
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# ĐÃ SỬA Ở ĐÂY: Thay model cũ bị xóa bằng Llama 3 70B
# ==========================================
MODEL_NAME = "llama3-70b-8192"

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        text_dau_vao = request.text
        
        # ==========================================
        # BƯỚC 1: DỊCH THÔ (Đã ép khuôn Tiếng Việt)
        # ==========================================
        chat_1 = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là biên dịch viên. TUYỆT ĐỐI CHỈ DỊCH SANG TIẾNG VIỆT sát nghĩa gốc nhất. CẤM SỬ DỤNG TIẾNG TRUNG (HÁN TỰ) HAY TIẾNG ANH."},
                {"role": "user", "content": text_dau_vao}
            ],
            model=MODEL_NAME,
            temperature=0.3 # Thêm cái này để AI bớt "ảo", tập trung dịch đúng từ
        )
        ban_dich_tho = chat_1.choices[0].message.content
        
        # ==========================================
        # BƯỚC 2: TRAU CHUỐT (Đã khóa mõm Hán tự)
        # ==========================================
        prompt_refine = f"""
        Viết lại bản dịch thô sau thành văn phong Light Novel/Tiên hiệp mượt mà.
        Giữ nguyên ý nghĩa, không cắt bớt chi tiết. CHỈ TRẢ VỀ BẢN DỊCH HOÀN CHỈNH BẰNG TIẾNG VIỆT, không giải thích gì thêm.
        TUYỆT ĐỐI KHÔNG ĐƯỢC XUẤT HIỆN HÁN TỰ.
        
        Bản gốc: {text_dau_vao}
        Bản dịch thô: {ban_dich_tho}
        """
        
        chat_2 = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là tác giả tiểu thuyết mạng nổi tiếng. Chỉ viết bằng TIẾNG VIỆT."},
                {"role": "user", "content": prompt_refine}
            ],
            model=MODEL_NAME,
            temperature=0.4
        )
        
        return {"translated_text": chat_2.choices[0].message.content}
        
    except Exception as e:
        error_msg = str(e)
        # BẮT LỖI NHẤN NHIỀU LẦN (RATE LIMIT / TOKEN LIMIT) TRẢ VỀ APP
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return {"translated_text": "⚠️ LỖI: Bạn đang bấm dịch quá nhanh hoặc đoạn văn quá dài gây kẹt API! Vui lòng đợi khoảng 1 phút rồi bấm lại nhé."}
        else:
            return {"translated_text": f"⚠️ LỖI HỆ THỐNG: {error_msg}"}
