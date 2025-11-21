import os
import json
import re
from pathlib import Path
from datetime import datetime

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

CATALOG_PATH    = DATA_DIR / "welllab_catalog.json"      # 25 combo
SYMPTOMS_PATH   = DATA_DIR / "symptoms_mapping.json"     # intent -> combo
FAQ_PATH        = DATA_DIR / "faq.json"                  # câu hỏi thường gặp
OBJECTIONS_PATH = DATA_DIR / "objections.json"           # từ chối phổ biến
USERS_PATH      = DATA_DIR / "users_store.json"          # hồ sơ người dùng

WELLLAB_CATALOG = load_json(CATALOG_PATH, [])
SYMPTOM_RULES   = load_json(SYMPTOMS_PATH, [])
FAQ_LIST        = load_json(FAQ_PATH, [])
OBJECTION_LIST  = load_json(OBJECTIONS_PATH, [])

def load_users_store():
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users_store(store: dict):
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Lỗi lưu users_store.json:", e)

USERS_STORE = load_users_store()

# ========= LOG HỘI THOẠI =========
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CONV_LOG_PATH = LOG_DIR / "conversations.log"

def get_now_iso():
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat()

def log_event(user_id: int, direction: str, text: str, extra: dict | None = None):
    """
    Ghi 1 dòng JSON vào logs/conversations.log
    direction: 'user' | 'bot'
    """
    rec = {
        "ts": get_now_iso(),
        "user_id": user_id,
        "direction": direction,
        "text": text
    }
    if extra:
        rec["meta"] = extra
    try:
        with open(CONV_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Lỗi ghi log hội thoại:", e)

# ========= HỒ SƠ NGƯỜI DÙNG (USER STORE) =========

def get_or_create_user_profile(telegram_user_id: int, tg_user: dict) -> dict:
    """
    Lấy hồ sơ user từ USERS_STORE hoặc tạo mới.
    tg_user: message.get("from")
    """
    uid = str(telegram_user_id)
    profile = USERS_STORE.get(uid) or {
        "telegram_id": telegram_user_id,
        "first_seen": get_now_iso(),
        "last_seen": get_now_iso(),
        "name": "",
        "username": "",
        "main_needs": {},       # đếm số lần hỏi theo need: health/product/policy/other
        "intents_count": {},    # đếm số lần theo intent: blood_pressure, liver,...
        "total_messages": 0,
        "notes": ""
    }

    # Cập nhật thông tin Telegram cơ bản
    if tg_user:
        uname = (tg_user.get("username") or "").strip()
        fname = (tg_user.get("first_name") or "").strip()
        lname = (tg_user.get("last_name") or "").strip()
        full_name = (fname + " " + lname).strip()
        if full_name:
            profile["name"] = full_name
        if uname:
            profile["username"] = uname

    profile["last_seen"] = get_now_iso()
    USERS_STORE[uid] = profile
    return profile

def touch_user_stats(profile: dict, need: str | None = None, intent: str | None = None):
    """Cập nhật thống kê hành vi vào profile (không gọi AI)."""
    profile["total_messages"] = int(profile.get("total_messages") or 0) + 1

    if need:
        needs = profile.get("main_needs") or {}
        needs[need] = int(needs.get(need) or 0) + 1
        profile["main_needs"] = needs

    if intent:
        intents = profile.get("intents_count") or {}
        intents[intent] = int(intents.get(intent) or 0) + 1
        profile["intents_count"] = intents

    # Lưu lại xuống file (đơn giản, mỗi request 1 lần vẫn ổn)
    save_users_store(USERS_STORE)

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
#   "profile": { ... },
#   "stage": "start" | "clarify" | "advise",
#   "first_issue": str | None,
#   "need": "health" | "product" | "policy" | "other"
# }

# ========= PROMPT HỆ THỐNG =========
BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn sức khỏe & thực phẩm bảo vệ sức khỏe WELLLAB cho công ty Con Đường Xanh.\n"
    "- Luôn coi sức khỏe và lợi ích lâu dài của khách hàng là trung tâm.\n"
    "- Luôn giải thích rõ ràng, dễ hiểu, không hù dọa, không hứa hẹn quá mức.\n"
    "- Chỉ dựa trên thông tin combo/sản phẩm được cung cấp trong ngữ cảnh, không bịa ra sản phẩm mới.\n"
    "- Không thay thế chẩn đoán hay đơn thuốc của bác sĩ, luôn khuyến nghị khách tham khảo bác sĩ khi cần.\n"
    "- Ưu tiên giúp khách hiểu vấn đề và định hướng lối sống, sau đó mới nhẹ nhàng gợi ý combo/sản phẩm phù hợp.\n"
)

TVV_SYSTEM_EXTRA = (
    "Ngữ cảnh: Người đang trao đổi với bạn là TƯ VẤN VIÊN của công ty, không phải khách hàng.\n"
    "- Hãy trả lời như đang huấn luyện nội bộ: giải thích combo, gợi ý cách tư vấn, cách xử lý thắc mắc.\n"
)

# ========= HÀM GỬI TIN =========

def send_message(chat_id: int, text: str):
    """Gửi tin nhắn về Telegram + ghi log bot."""
    try:
        # Ghi log bot
        log_event(chat_id, "bot", text, extra={"source": "bot_reply"})
    except Exception as e:
        print("Lỗi log bot:", e)

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
        s = {
            "mode": "customer",
            "intent": None,
            "profile": {},
            "stage": "start",
            "first_issue": None,
            "need": None  # 'health' | 'product' | 'policy' | 'other'
        }
        SESSIONS[chat_id] = s
    return s

# ========= NHẬN DIỆN INTENT & NEED =========

INTENT_PRIORITY_DEFAULT = 10  # fallback

def get_intent_priority(intent: str) -> int:
    for rule in SYMPTOM_RULES:
        if rule.get("intent") == intent:
            return int(rule.get("priority", INTENT_PRIORITY_DEFAULT))
    return INTENT_PRIORITY_DEFAULT

def detect_intent_from_text(text: str) -> str | None:
    """
    Phát hiện intent dựa trên bảng symptoms_mapping.json.
    - Mỗi từ khóa khớp +1 điểm.
    - Điểm cuối = matches * 10 + priority (ưu tiên huyết áp, tiểu đường, gan... cao hơn).
    - Trả về intent có điểm cao nhất nếu có ít nhất 1 từ khóa khớp.
    """
    t = text.lower()
    best_intent = None
    best_score = 0

    for rule in SYMPTOM_RULES:
        intent = rule.get("intent")
        kws = rule.get("keywords", [])
        matches = 0
        for kw in kws:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue
            if kw_l in t:
                matches += 1

        if matches > 0:
            priority = get_intent_priority(intent)
            score = matches * 10 + priority  # nhân trọng số + ưu tiên
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent

def detect_need(text: str) -> str:
    """
    Xác định khách đang quan tâm chính là gì:
    - 'health': triệu chứng, bệnh, đau ở đâu...
    - 'product': hỏi về combo, sản phẩm, thành phần, giá...
    - 'policy': hỏi về mua hàng, giao hàng, thanh toán, đổi trả...
    - 'other': còn lại
    """
    t = text.lower()

    health_kws = [
        "đau ", "bị đau", "bệnh", "trị bệnh", "triệu chứng", "huyết áp", "tiểu đường",
        "mỡ máu", "gan", "thận", "da cơ địa", "vảy nến", "mất ngủ", "khó ngủ", "ho", "khó thở",
        "viêm", "ngứa", "mụn"
    ]
    product_kws = [
        "sản phẩm", "combo", "liệu trình", "loại nào", "dùng gì",
        "công dụng", "thành phần", "uống như thế nào", "cách dùng", "bao lâu",
        "giá bao nhiêu", "bao nhiêu tiền"
    ]
    policy_kws = [
        "mua hàng", "đặt hàng", "mua ở đâu", "ship", "giao hàng",
        "thanh toán", "chuyển khoản", "cod", "đổi trả", "bảo hành", "chính sách"
    ]

    if any(kw in t for kw in health_kws):
        return "health"
    if any(kw in t for kw in product_kws):
        return "product"
    if any(kw in t for kw in policy_kws):
        return "policy"
    return "other"

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
        except Exception:
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

# ========= CÂU HỎI LÀM RÕ THEO INTENT =========

CLARIFY_QUESTIONS = {
    "blood_pressure": (
        "Để em tư vấn chính xác hơn về *huyết áp*, anh/chị cho em hỏi thêm một chút nhé:\n"
        "- Anh/chị bị cao huyết áp lâu chưa, đã được bác sĩ chẩn đoán hay tự đo ở nhà ạ?\n"
        "- Hiện tại có đang dùng thuốc huyết áp đều đặn không?\n"
        "- Anh/chị có kèm theo triệu chứng như đau đầu, chóng mặt, khó thở hay đau ngực không?"
    ),
    "diabetes": (
        "Về *tiểu đường*, để tư vấn rõ hơn anh/chị giúp em:\n"
        "- Anh/chị được chẩn đoán tiểu đường type mấy và bao lâu rồi ạ?\n"
        "- Đường huyết gần đây đo được khoảng bao nhiêu?\n"
        "- Anh/chị có đang dùng thuốc hay tiêm insulin không?"
    ),
    "weight_loss": (
        "Về vấn đề *thừa cân, béo phì*, anh/chị cho em biết thêm:\n"
        "- Chiều cao, cân nặng hiện tại khoảng bao nhiêu?\n"
        "- Anh/chị tăng cân lâu chưa và có từng giảm cân nhưng bị tăng lại không?\n"
        "- Hiện tại chế độ ăn uống và vận động của anh/chị như thế nào (ít vận động/nhiều tinh bột...)?"
    ),
    "digestive": (
        "Về *tiêu hoá*, anh/chị chia sẻ rõ hơn giúp em nhé:\n"
        "- Anh/chị hay bị đầy bụng, ợ hơi, ợ chua hay táo bón/tiêu chảy?\n"
        "- Triệu chứng kéo dài bao lâu rồi và có từng nội soi hay khám dạ dày/chức năng tiêu hoá chưa?\n"
        "- Ăn uống có thất thường, bỏ bữa hoặc dùng nhiều rượu bia, cà phê không?"
    ),
    "respiratory": (
        "Về *hô hấp*, anh/chị mô tả thêm giúp em:\n"
        "- Anh/chị hay ho khan, ho có đờm hay khó thở, khò khè?\n"
        "- Triệu chứng kéo dài bao lâu, có thường xuyên tái lại theo mùa không?\n"
        "- Anh/chị có hút thuốc hoặc làm việc trong môi trường khói bụi không?"
    ),
    "skin_psoriasis": (
        "Về *viêm da cơ địa/vảy nến*, anh/chị giúp em vài thông tin nhé:\n"
        "- Tình trạng da hiện tại: đỏ rát, bong vảy, ngứa nhiều hay chỉ khô nứt ạ?\n"
        "- Vùng da bị ở tay, chân, thân mình hay lan rộng khắp người?\n"
        "- Anh/chị đã từng dùng thuốc bôi/uống của bác sĩ da liễu chưa, và có bệnh nền dị ứng nào không?"
    ),
    # fallback chung cho các intent khác
    "default": (
        "Để em hiểu rõ hơn và tư vấn đúng, anh/chị cho em biết thêm:\n"
        "- Triệu chứng chính anh/chị đang gặp là gì và kéo dài bao lâu rồi?\n"
        "- Anh/chị bao nhiêu tuổi, giới tính gì và có bệnh nền/đang dùng thuốc gì không?\n"
        "- Mục tiêu của anh/chị là giảm triệu chứng, phòng tái phát hay nâng tổng thể sức khoẻ ạ?"
    )
}

def get_clarify_question(intent: str | None) -> str:
    if not intent:
        return CLARIFY_QUESTIONS["default"]
    return CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["default"])

# ========= GỌI OPENAI =========

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

    # Lấy user_id & hồ sơ người dùng
    tg_user = message.get("from") or {}
    user_id = tg_user.get("id", chat_id)  # thường giống nhau trong chat riêng
    profile = get_or_create_user_profile(user_id, tg_user)

    # Ghi log tin nhắn của khách
    log_event(
        user_id,
        "user",
        text_stripped,
        extra={"username": profile.get("username"), "name": profile.get("name")}
    )

    session = get_session(chat_id)

    # ----- LỆNH CƠ BẢN -----
    if text_stripped.startswith("/start"):
        session["mode"] = "customer"
        session["intent"] = None
        session["profile"] = {}
        session["stage"] = "start"
        session["first_issue"] = None
        session["need"] = None

        welcome = (
            "Chào anh/chị 👋\n"
            "Em là trợ lý AI hỗ trợ tư vấn & chăm sóc sức khỏe bằng sản phẩm WELLLAB.\n\n"
            "Để em hỗ trợ đúng hơn, anh/chị đang quan tâm nhất đến điều gì ạ?\n"
            "• Tình trạng sức khỏe hiện tại (đau, bệnh, triệu chứng…)\n"
            "• Thông tin sản phẩm/combo WELLLAB\n"
            "• Chính sách mua hàng, giao hàng, thanh toán\n"
            "Hoặc anh/chị cứ nhắn tự nhiên, em sẽ lắng nghe và hỗ trợ ạ. 💚"
        )
        send_message(chat_id, welcome)
        return "ok", 200

    if text_stripped.lower() == "/tvv":
        session["mode"] = "tvv"
        send_message(
            chat_id,
            "Đã chuyển sang *chế độ TƯ VẤN VIÊN*. Anh/chị có thể hỏi về combo, sản phẩm hoặc cách tư vấn cho khách."
        )
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

    # ----- THỬ TRẢ LỜI FAQ (không tốn token) -----
    faq_answer = try_answer_faq(text_stripped)
    if faq_answer:
        send_message(chat_id, faq_answer)
        # cập nhật thống kê nhu cầu (không cần intent)
        need = session.get("need") or detect_need(text_stripped)
        session["need"] = need
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ----- THỬ XỬ LÝ TỪ CHỐI (không tốn token) -----
    obj_answer = try_answer_objection(text_stripped)
    if obj_answer:
        send_message(chat_id, obj_answer)
        need = session.get("need") or detect_need(text_stripped)
        session["need"] = need
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== XÁC ĐỊNH NHU CẦU CHÍNH (NEED) LẦN ĐẦU ======
    if not session.get("need"):
        session["need"] = detect_need(text_stripped)

    need = session.get("need") or "other"

    # Nếu nhu cầu chính là CHÍNH SÁCH / MUA HÀNG -> ưu tiên trả lời ngắn gọn, không cần intent
    if need == "policy":
        faq_answer = try_answer_faq(text_stripped)
        if faq_answer:
            send_message(chat_id, faq_answer)
            touch_user_stats(profile, need=need, intent=None)
            return "ok", 200

        combo = None  # không cần combo
        reply = call_openai_for_answer(
            "Khách đang hỏi về CHÍNH SÁCH hoặc MUA HÀNG. "
            "Hãy trả lời ngắn gọn, rõ ràng. Không tư vấn bệnh hoặc liệu trình.\n\n"
            "Câu hỏi của khách: " + text_stripped,
            session,
            combo
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # Nếu nhu cầu chính là THÔNG TIN SẢN PHẨM mà chưa có intent rõ
    if need == "product" and not detect_intent_from_text(text_stripped):
        ask = (
            "Dạ, anh/chị muốn tìm hiểu về *sản phẩm/combo* nào của WELLLAB ạ?\n"
            "Anh/chị có thể gửi tên combo, mã số trên tài liệu hoặc mục tiêu chính "
            "(ví dụ: giảm mỡ, hỗ trợ gan, viêm da cơ địa...)."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== FLOW “TƯ VẤN VIÊN Y TẾ CÓ TÂM” ======

    # 1. Cập nhật / phát hiện intent mới
    new_intent = detect_intent_from_text(text_stripped)
    if new_intent:
        session["intent"] = new_intent

    intent = session.get("intent")
    stage = session.get("stage", "start")

    # Cập nhật thống kê sau khi đã có need & (có thể) intent
    touch_user_stats(profile, need=need, intent=intent)

    # 2. Nếu CHƯA có intent rõ ràng -> hỏi chung để hiểu thêm (không gợi ý combo)
    if not intent:
        question = get_clarify_question(None)
        session["stage"] = "clarify"
        if not session.get("first_issue"):
            session["first_issue"] = text_stripped
        send_message(chat_id, question)
        return "ok", 200

    # 3. Nếu có intent nhưng đang ở giai đoạn START -> hỏi làm rõ theo intent (không gợi ý combo)
    if stage in ("start", None):
        session["first_issue"] = text_stripped
        session["stage"] = "clarify"
        question = get_clarify_question(intent)
        send_message(chat_id, question)
        return "ok", 200

    # 4. Nếu đang ở giai đoạn CLARIFY -> đã có đủ thông tin để tư vấn lần đầu
    if stage == "clarify":
        issue = session.get("first_issue") or ""
        combined_user_text = (
            "Mô tả ban đầu của khách: " + issue + "\n\n"
            "Thông tin bổ sung khách vừa cung cấp: " + text_stripped
        )

        combo = choose_combo(intent)
        session["stage"] = "advise"   # đã tư vấn lần đầu, từ sau là follow-up
        reply = call_openai_for_answer(combined_user_text, session, combo)
        send_message(chat_id, reply)
        return "ok", 200

    # 5. Nếu đã ở giai đoạn ADVISE -> coi là câu hỏi bổ sung trong cùng chủ đề
    if stage == "advise":
        combo = choose_combo(intent)
        reply = call_openai_for_answer(text_stripped, session, combo)
        send_message(chat_id, reply)
        return "ok", 200

    # Fallback an toàn nếu vì lý do nào đó không vào nhánh nào ở trên
    combo = choose_combo(intent)
    reply = call_openai_for_answer(text_stripped, session, combo)
    send_message(chat_id, reply)
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
