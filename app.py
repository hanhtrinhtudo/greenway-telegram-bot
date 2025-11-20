import os
import json
import re
from pathlib import Path

import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# ========= ĐƯỜNG DẪN & DATA =========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Không load được {path}: {e}")
        return default

CATALOG_PATH   = DATA_DIR / "welllab_catalog.json"      # 25 combo
SYMPTOMS_PATH  = DATA_DIR / "symptoms_mapping.json"     # intent -> combo
FAQ_PATH       = DATA_DIR / "faq.json"                  # câu hỏi thường gặp
OBJECTIONS_PATH= DATA_DIR / "objections.json"           # từ chối phổ biến

WELLLAB_CATALOG = load_json(CATALOG_PATH, [])
SYMPTOM_RULES   = load_json(SYMPTOMS_PATH, [])
FAQ_LIST        = load_json(FAQ_PATH, [])
OBJECTION_LIST  = load_json(OBJECTIONS_PATH, [])

# ========= TELEGRAM & OPENAI =========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Chưa cấu hình TELEGRAM_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Chưa cấu hình OPENAI_API_KEY")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= SESSION LƯU THEO CHAT =========
SESSIONS = {}
# SESSIONS[chat_id] = {
#   "mode": "customer" | "tvv",
#   "intent": str | None,
#   "profile": { ... }
# }

# ========= PROMPT HỆ THỐNG =========
BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn sức khỏe & thực phẩm bảo vệ sức khỏe WELLLAB cho công ty Con Đường Xanh.\n"
    "- Trả lời bằng tiếng Việt, xưng hô anh/chị – em.\n"
    "- Chỉ dựa trên thông tin combo/sản phẩm được cung cấp trong ngữ cảnh.\n"
    "- Không bịa ra sản phẩm mới, không tự thêm công dụng y khoa.\n"
    "- Không thay thế chẩn đoán hay đơn thuốc của bác sĩ.\n"
)

TVV_SYSTEM_EXTRA = (
    "Ngữ cảnh: Người đang trao đổi với bạn là TƯ VẤN VIÊN của công ty, không phải khách hàng.\n"
    "- Hãy trả lời như đang huấn luyện nội bộ: giải thích combo, gợi ý cách tư vấn, cách xử lý thắc mắc.\n"
)

# ========= HÀM GỬI TIN =========

def send_message(chat_id: int, text: str):
    try:
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
    except Exception as e:
        print("Lỗi gửi message về Telegram:", e)

# ========= SESSION =========

def get_session(chat_id: int) -> dict:
    s = SESSIONS.get(chat_id)
    if not s:
        s = {"mode": "customer", "intent": None, "profile": {}}
        SESSIONS[chat_id] = s
    return s

# ========= NHẬN DIỆN INTENT & PROFILE =========

def detect_intent_from_text(text: str) -> str | None:
    t = text.lower()
    best_intent = None
    best_score = 0
    for rule in SYMPTOM_RULES:
        score = 0
        for kw in rule.get("keywords", []):
            if kw.lower() in t:
                score += 1
        if score > best_score and score > 0:
            best_score = score
            best_intent = rule.get("intent")
    return best_intent

def choose_combo(intent: str | None) -> dict | None:
    if not intent:
        return None
    rule = next((r for r in SYMPTOM_RULES if r.get("intent") == intent), None)
    if not rule:
        return None
    preferred_names = rule.get("preferred_combos", [])
    for name in preferred_names:
        combo = next((c for c in WELLLAB_CATALOG if c.get("name") == name), None)
        if combo:
            return combo
    return None

def extract_profile(text: str) -> dict:
    profile = {}
    lower = text.lower()

    m_age = re.search(r"(\d{2})\s*t[uô]i", lower)
    if m_age:
        try:
            profile["age"] = int(m_age.group(1))
        except:
            pass

    if "nam" in lower:
        profile["gender"] = "nam"
    if "nữ" in lower or "nu" in lower:
        profile["gender"] = "nữ"

    if "không bệnh nền" in lower or "ko bệnh nền" in lower or "k bệnh nền" in lower:
        profile["has_chronic"] = False
    elif "bệnh nền" in lower:
        profile["has_chronic"] = True

    return profile

# ========= FAQ & OBJECTION MATCHING (KHÔNG GỌI AI) =========

def match_keywords_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return True
    return False

def try_answer_faq(text: str) -> str | None:
    for item in FAQ_LIST:
        kws = item.get("keywords_any", [])
        if kws and match_keywords_any(text, kws):
            return item.get("answer")
    return None

def try_answer_objection(text: str) -> str | None:
    for item in OBJECTION_LIST:
        kws = item.get("keywords_any", [])
        if kws and match_keywords_any(text, kws):
            return item.get("answer")
    return None

# ========= XÂY CONTEXT GỬI OPENAI =========

def build_combo_context(combo: dict | None) -> str:
    if not combo:
        return "Hiện chưa xác định được combo cụ thể."

    lines = []
    lines.append(f"Combo: {combo.get('name','')}")
    header = combo.get("header_text", "")
    if header:
        lines.append("\n[Thông tin]:")
        lines.append(header)

    duration = combo.get("duration_text", "")
    if duration:
        lines.append("\n[Thời gian liệu trình khuyến nghị]:")
        lines.append(duration)

    prods = combo.get("products", [])
    if prods:
        lines.append("\n[Thành phần]:")
        for idx, p in enumerate(prods, start=1):
            lines.append(f"{idx}. {p.get('name','')}: {p.get('text','')}")
    return "\n".join(lines)

def build_profile_context(profile: dict) -> str:
    if not profile:
        return "Chưa có thêm thông tin cụ thể về tuổi, giới tính hay bệnh nền."
    parts = []
    if profile.get("age"):
        parts.append(f"Tuổi khoảng: {profile['age']}.")
    if profile.get("gender"):
        parts.append(f"Giới tính: {profile['gender']}.")
    if profile.get("has_chronic") is True:
        parts.append("Có bệnh nền (chi tiết chưa rõ).")
    elif profile.get("has_chronic") is False:
        parts.append("Không có bệnh nền.")
    return " ".join(parts)

def call_openai_for_answer(user_text: str, session: dict, combo: dict | None) -> str:
    mode = session.get("mode", "customer")
    intent = session.get("intent")
    profile = session.get("profile", {})

    sys_prompt = BASE_SYSTEM_PROMPT
    if mode == "tvv":
        sys_prompt += "\n" + TVV_SYSTEM_EXTRA

    combo_ctx = build_combo_context(combo)
    profile_ctx = build_profile_context(profile)
    intent_text = f"Intent hiện tại: {intent or 'chưa rõ'}."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "system",
                    "content": (
                        "Dữ liệu nội bộ:\n"
                        + intent_text + "\n\n"
                        + "[HỒ SƠ KHÁCH]: " + profile_ctx + "\n\n"
                        + "[COMBO LIÊN QUAN]:\n" + combo_ctx
                    )
                },
                {"role": "user", "content": user_text}
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Lỗi gọi OpenAI:", e)
        return "Hiện hệ thống AI đang bận, anh/chị vui lòng thử lại sau một chút nhé."

# ========= XỬ LÝ CÂU CHÀO ĐƠN GIẢN =========

def is_simple_greeting(text: str) -> bool:
    t = text.lower().strip()
    simple = ["chào", "chào em", "hi", "hello", "alo", "ok", "oke", "cảm ơn", "thanks", "thank you"]
    return any(t.startswith(s) or t == s for s in simple)

def greeting_reply(text: str) -> str:
    t = text.lower()
    if "cảm ơn" in t or "thanks" in t or "thank" in t:
        return "Em cảm ơn anh/chị ạ 😊 Nếu còn câu hỏi nào về sản phẩm hay liệu trình, anh/chị cứ nhắn cho em nhé."
    return "Em chào anh/chị 👋 Anh/chị đang quan tâm tới vấn đề sức khỏe nào để em hỗ trợ ạ?"

# ========= ROUTES =========

@app.route("/", methods=["GET"])
def index():
    return "Bot is running.", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    print("Update:", update)

    message = update.get("message")
    if not message:
        return "no message", 200

    chat_id = message["chat"]["id"]
    text = message.get("text") or ""
    text_stripped = text.strip()

    session = get_session(chat_id)

    # ----- LỆNH CƠ BẢN -----
    if text_stripped.startswith("/start"):
        session["mode"] = "customer"
        session["intent"] = None
        session["profile"] = {}
        welcome = (
            "Chào anh/chị 👋\n"
            "Em là trợ lý AI hỗ trợ tư vấn & chăm sóc sức khỏe bằng sản phẩm WELLLAB.\n"
            "Anh/chị cứ gửi nhu cầu, triệu chứng hoặc câu hỏi về sản phẩm, liệu trình… để em hỗ trợ nhé."
        )
        send_message(chat_id, welcome)
        return "ok", 200

    if text_stripped.lower() == "/tvv":
        session["mode"] = "tvv"
        send_message(chat_id, "Đã chuyển sang *chế độ TƯ VẤN VIÊN*. Anh/chị có thể hỏi về combo, sản phẩm hoặc cách tư vấn cho khách.")
        return "ok", 200

    if text_stripped.lower() == "/kh":
        session["mode"] = "customer"
        send_message(chat_id, "Đã chuyển về *chế độ tư vấn khách hàng*.")
        return "ok", 200

    # ----- CÂU CHÀO ĐƠN GIẢN → TRẢ LỜI CỐ ĐỊNH -----
    if is_simple_greeting(text_stripped):
        send_message(chat_id, greeting_reply(text_stripped))
        return "ok", 200

    # ----- CẬP NHẬT PROFILE (KHÔNG DÙNG AI) -----
    prof_update = extract_profile(text_stripped)
    if prof_update:
        session["profile"] = {**session.get("profile", {}), **prof_update}

    # ----- THỬ TRẢ LỜI FAQ -----
    faq_answer = try_answer_faq(text_stripped)
    if faq_answer:
        send_message(chat_id, faq_answer)
        return "ok", 200

    # ----- THỬ XỬ LÝ TỪ CHỐI -----
    obj_answer = try_answer_objection(text_stripped)
    if obj_answer:
        send_message(chat_id, obj_answer)
        return "ok", 200

    # ----- XÁC ĐỊNH / GIỮ INTENT -----
    if session.get("intent") is None:
        session["intent"] = detect_intent_from_text(text_stripped)

    intent = session.get("intent")
    combo = choose_combo(intent)

    # ----- GỌI OPENAI (CHỈ KHI THỰC SỰ CẦN) -----
    reply = call_openai_for_answer(text_stripped, session, combo)
    send_message(chat_id, reply)

    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
