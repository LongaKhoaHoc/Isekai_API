from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import re
from groq import Groq

app = FastAPI()

# Lấy API Key từ môi trường
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"

# Input Schema cho API
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Đoạn tiếng Nhật cần dịch")
    context: str = Field(default="", description="Bối cảnh (VD: Nam chính đang nói chuyện với Ma Vương)")
    glossary: dict = Field(default={}, description="Từ điển tên riêng (VD: {'リグルド': 'Rigurd'})")

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        text_dau_vao = request.text
        context = request.context
        
        # Xử lý từ điển tên riêng
        glossary_str = ", ".join([f"{k} -> {v}" for k, v in request.glossary.items()]) if request.glossary else "Không có"

        # ==========================================
        # BƯỚC 1: PHÂN TÍCH LOGIC & DỊCH THÔ (Chain of Thought)
        # ==========================================
        # Ép AI phân tích ngữ pháp, xưng hô trước khi dịch để chống sai lệch nghĩa
        prompt_1 = f"""
        Bối cảnh hội thoại: {context if context else 'Không rõ'}
        Từ điển bắt buộc: {glossary_str}
        
        Văn bản gốc:
        {text_dau_vao}
        
        Hãy thực hiện 2 bước sau:
        1. Phân tích cấu trúc câu, xác định chủ ngữ bị ẩn (ai nói với ai), và ý nghĩa gốc. Viết vào trong thẻ <thinking>.
        2. Dịch thô sát nghĩa nhất sang tiếng Việt (giữ nguyên kính ngữ -san, -sama). Viết vào trong thẻ <translation>.
        """

        chat_1 = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích ngôn ngữ tiếng Nhật. Luôn tuân thủ định dạng thẻ XML được yêu cầu."},
                {"role": "user", "content": prompt_1}
            ],
            model=MODEL_NAME,
            temperature=0.1 # Nhiệt độ cực thấp để đảm bảo logic và dịch sát nghĩa
        )
        
        response_1 = chat_1.choices[0].message.content
        
        # Dùng Regex để tách lấy phần bản dịch thô nằm trong thẻ <translation>
        match = re.search(r'<translation>(.*?)</translation>', response_1, re.DOTALL)
        if match:
            ban_dich_tho = match.group(1).strip()
        else:
            ban_dich_tho = response_1 # Fallback nếu AI không trả về thẻ XML

        # ==========================================
        # BƯỚC 2: TRAU CHUỐT VĂN PHONG LIGHT NOVEL (Polishing)
        # ==========================================
        prompt_2 = f"""
        Bản gốc tiếng Nhật: {text_dau_vao}
        Bản dịch thô: {ban_dich_tho}
        Bối cảnh: {context if context else 'Tự suy luận'}
        Từ điển: {glossary_str}
        
        Nhiệm vụ: Dựa vào bản dịch thô, hãy viết lại thành văn phong Light Novel / Isekai mượt mà.
        Yêu cầu:
        1. Xưng hô chuẩn xác (hắn, y, thiếu niên, cô gái, cậu...).
        2. Cảm xúc, tự nhiên, bỏ sự khô khan của máy dịch.
        3. TUYỆT ĐỐI KHÔNG BỊA THÊM CHI TIẾT hay giải thích lằng nhằng. CHỈ IN RA KẾT QUẢ CUỐI CÙNG.
        """
        
        chat_2 = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là một Biên tập viên Light Novel lão làng người Việt Nam. Nhiệm vụ duy nhất của bạn là xuất ra văn bản tiếng Việt mượt mà."},
                {"role": "user", "content": prompt_2}
            ],
            model=MODEL_NAME,
            temperature=0.4 # Tăng nhẹ nhiệt độ để văn phong bay bổng, có tính nghệ thuật
        )
        
        final_translation = chat_2.choices[0].message.content.strip()
        
        return {
            "translated_text": final_translation,
            # Trả về thêm bản dịch thô để debug xem AI có dịch sai ở Bước 1 không (Bro có thể xóa dòng này đi nếu không cần)
            "_debug_rough_translation": ban_dich_tho 
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="⚠️ LỖI: Groq đang bị Rate Limit. Đợi 30s rồi thử lại.")
        else:
            raise HTTPException(status_code=500, detail=f"⚠️ LỖI HỆ THỐNG: {error_msg}")
