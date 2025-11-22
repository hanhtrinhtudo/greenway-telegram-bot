import os
import json
import re
import unicodedata
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
    """Đọc file JSON, nếu lỗi trả về default để bot vẫn chạy được."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Không load được {path}: {e}")
        return default


CATALOG_PATH = DATA_DIR / "welllab_catalog.json"      # danh mục combo
SYMPTOMS_PATH = DATA_DIR / "symptoms_mapping.json"    # intent -> combo
FAQ_PATH = DATA_DIR / "faq.json"                      # câu hỏi thường gặp
OBJECTIONS_PATH = DATA_DIR / "objections.json"        # từ chối phổ biến
USERS_PATH = DATA_DIR / "users_store.json"            # hồ sơ người dùng

WELLLAB_CATALOG = load_json(CATALOG_PATH, [])
SYMPTOM_RULES = load_json(SYMPTOMS_PATH, [])
FAQ_LIST = load_json(FAQ_PATH, [])
OBJECTION_LIST = load_json(OBJECTIONS_PATH, [])


# ========= TIỆN ÍCH CHUẨN HÓA =========
def normalize_text(s: str) -> str:
    """Bỏ dấu, về thường để so khớp tên combo linh hoạt hơn."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def search_combo_by_text(query: str, top_k: int = 1) -> list[dict]:
    """
    Tìm combo theo tên / alias trong welllab_catalog.json.
    So khớp không dấu, không phân biệt hoa thường.
    """
    q = normalize_text(query)
    if not q or not WELLLAB_CATALOG:
        return []

    results: list[tuple[int, dict]] = []
    for combo in WELLLAB_CATALOG:
        name = normalize_text(combo.get("name", ""))
        aliases = [normalize_text(a) for a in combo.get("aliases", [])]
        haystack = " ".join([name] + aliases)

        score = 0
        for token in q.split():
            if token and token in haystack:
                score += 1

        if score > 0:
            results.append((score, combo))

    results.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in results[:top_k]]


# ========= USER STORE =========
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
    rec: dict = {
        "ts": get_now_iso(),
        "user_id": user_id,
        "direction": direction,
        "text": text,
    }
    if extra:
        rec["meta"] = extra
    try:
        with open(CONV_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Lỗi ghi log hội thoại:", e)


# ========= HỒ SƠ NGƯỜI DÙNG =========
def get_or_create_user_profile(telegram_user_id: int, tg_user: dict) -> dict:
    uid = str(telegram_user_id)
    profile = USERS_STORE.get(uid) or {
        "telegram_id": telegram_user_id,
        "first_seen": get_now_iso(),
        "last_seen": get_now_iso(),
        "name": "",
        "username": "",
        "main_needs": {},
        "intents_count": {},
        "total_messages": 0,
        "notes": "",
    }

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
    profile["total_messages"] = int(profile.get("total_messages") or 0) + 1

    if need:
        needs = profile.get("main_needs") or {}
        needs[need] = int(needs.get(need) or 0) + 1
        profile["main_needs"] = needs

    if intent:
        intents = profile.get("intents_count") or {}
        intents[intent] = int(intents.get(intent) or 0) + 1
        profile["intents_count"] = intents

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

# ========= SESSION THEO CHAT =========
SESSIONS: dict[int, dict] = {}


def get_session(chat_id: int) -> dict:
    s = SESSIONS.get(chat_id)
    if not s:
        s = {
            "mode": "customer",
            "intent": None,
            "profile": {},
            "stage": "await_need",
            "first_issue": None,
            "need": None,
            "last_combo": None,  # combo đã tư vấn gần nhất
        }
        SESSIONS[chat_id] = s
    return s


# ========= PROMPT HỆ THỐNG =========
BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn sức khỏe & thực phẩm bảo vệ sức khỏe WELLLAB cho công ty Con Đường Xanh.\n"
    "MỤC TIÊU CHÍNH:\n"
    "- Đặt sức khỏe và lợi ích DÀI HẠN của khách lên trước bán hàng.\n"
    "- Luôn lắng nghe, hỏi lại cho rõ rồi mới gợi ý sản phẩm.\n"
    "- Trả lời NGẮN GỌN, dễ hiểu, đúng trọng tâm câu hỏi hiện tại.\n"
    "- Tuyệt đối không hù dọa, không hứa hẹn chữa khỏi bệnh, không nói quá công dụng.\n"
    "- Chỉ dùng đúng các combo/sản phẩm có trong ngữ cảnh, không bịa thêm.\n"
    "- Luôn nhắc đây là thực phẩm bảo vệ sức khỏe, không thay thế chẩn đoán/đơn thuốc; khi tình trạng nặng hoặc kéo dài phải gặp bác sĩ.\n\n"
    "PHONG CÁCH TƯ VẤN:\n"
    "- Bước 1: Ghi nhận vấn đề của khách, phản hồi bằng 1–2 câu đồng cảm, dùng xưng hô thân thiện (anh/chị).\n"
    "- Bước 2: Hỏi lại 2–3 câu NGẮN để làm rõ (thời gian bị, mức độ, tuổi, bệnh nền, thuốc đang dùng...).\n"
    "- Bước 3: Tóm tắt lại ngắn gọn rồi mới gợi ý combo/sản phẩm (nếu phù hợp).\n"
    "- Khi khách CHỈ hỏi một thông tin cụ thể (ví dụ: link sản phẩm, giá, cách uống), hãy trả lời đúng ý, càng ngắn càng tốt, KHÔNG lặp lại toàn bộ mô tả combo.\n"
    "- Mỗi lần trả lời tối đa khoảng 8–10 dòng chat, ưu tiên bullet gạch đầu dòng, tránh văn bản quá dài.\n"
    "- Luôn kết thúc bằng một câu hỏi mở rất ngắn (ví dụ: 'Anh/chị thấy như vậy ổn không ạ?' hoặc 'Anh/chị cần em giải thích thêm phần nào không?').\n"
)

TVV_SYSTEM_EXTRA = (
    "Ngữ cảnh: Người đang trao đổi với bạn là *TƯ VẤN VIÊN* của công ty, không phải khách hàng cuối.\n"
    "- Trả lời như đang huấn luyện nội bộ: giải thích mục tiêu từng combo, cách đặt câu hỏi, cách xử lý thắc mắc.\n"
    "- Luôn nhắc lại quy trình tư vấn 5 bước để tư vấn viên áp dụng với khách.\n"
)


# ========= LỜI CHÀO / MENU =========
def build_welcome_message() -> str:
    return (
        "Chào anh/chị 👋\n"
        "Em là trợ lý AI hỗ trợ tư vấn & chăm sóc sức khỏe bằng sản phẩm WELLLAB.\n\n"
        "Để em hỗ trợ ĐÚNG NHU CẦU, anh/chị cho em biết mình quan tâm nhất đến điều gì ạ:\n"
        "- *Sức khỏe hiện tại*: đau/bệnh/triệu chứng đang gặp phải\n"
        "- *Sản phẩm/combo*: muốn tìm hiểu công dụng, cách dùng, liệu trình\n"
        "- *Chính sách*: mua hàng, giao hàng, thanh toán, đổi trả\n\n"
        "Anh/chị có thể chọn trên menu hoặc nhắn ngắn gọn: *“Anh bị… muốn cải thiện…”* "
        "hoặc *“Anh muốn hỏi về combo…”* để em hỗ trợ ạ. 💚"
    )


def get_main_menu_keyboard():
    return [
        ["🩺 Tư vấn theo triệu chứng"],
        ["🧴 Tư vấn theo combo / sản phẩm"],
        ["📦 Hỏi chính sách mua hàng"],
    ]


# ========= INTENT & NEED =========
INTENT_PRIORITY_DEFAULT = 10


def get_intent_priority(intent: str) -> int:
    for rule in SYMPTOM_RULES:
        if rule.get("intent") == intent:
            return int(rule.get("priority", INTENT_PRIORITY_DEFAULT))
    return INTENT_PRIORITY_DEFAULT


def detect_intent_from_text(text: str) -> str | None:
    t = text.lower()
    best_intent = None
    best_score = 0

    for rule in SYMPTOM_RULES:
        intent = rule.get("intent")
        kws = rule.get("keywords", [])
        matches = 0
        for kw in kws:
            kw_l = kw.lower().strip()
            if kw_l and kw_l in t:
                matches += 1

        if matches > 0:
            priority = get_intent_priority(intent)
            score = matches * 10 + priority
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent


def detect_need(text: str) -> str:
    t = text.lower()

    health_kws = [
        "đau ", "bị đau", "benh", "bệnh", "triệu chứng", "huyết áp", "tiểu đường",
        "mỡ máu", "gan", "thận", "da cơ địa", "vảy nến", "mất ngủ", "khó ngủ", "ho", "khó thở",
        "viêm", "ngứa", "mụn", "sức khỏe", "suc khoe",
    ]
    product_kws = [
        "sản phẩm", "san pham", "combo", "liệu trình", "lieu trinh", "loại nào", "dùng gì",
        "công dụng", "thành phần", "uống như thế nào", "cách dùng", "bao lâu",
        "giá bao nhiêu", "bao nhiêu tiền",
    ]
    policy_kws = [
        "mua hàng", "dat hang", "đặt hàng", "mua ở đâu", "ship", "giao hàng",
        "thanh toán", "thanh toan", "chuyển khoản", "cod", "đổi trả", "bảo hành",
        "bao hanh", "chính sách",
    ]

    if any(kw in t for kw in health_kws):
        return "health"
    if any(kw in t for kw in product_kws):
        return "product"
    if any(kw in t for kw in policy_kws):
        return "policy"
    return "other"


# ========= KHÔNG CÓ VẤN ĐỀ SỨC KHOẺ =========
NO_HEALTH_PATTERNS = [
    "không", "ko", "k", "khong", "hong", "hông",
    "không có", "ko có", "k có",
    "không bị", "ko bị", "k bị",
    "không vấn đề", "k vấn đề", "ko vấn đề",
    "không sao", "ko sao", "k sao",
]


def is_no_health_intent(text: str) -> bool:
    t = text.lower().strip()
    if t in ["không", "ko", "k", "khong"]:
        return True
    for p in NO_HEALTH_PATTERNS:
        if t == p or t.startswith(p + " "):
            return True
    return False


# ========= CHỌN COMBO TỪ INTENT =========
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


# ========= TRÍCH HỒ SƠ TỪ VĂN BẢN =========
def extract_profile(text: str) -> dict:
    profile: dict = {}
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


# ========= FAQ & OBJECTIONS =========
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


# ========= CONTEXT GỬI OPENAI =========
def build_combo_context(combo: dict | None) -> str:
    if not combo:
        return "Hiện chưa xác định được combo cụ thể."

    lines: list[str] = []
    lines.append(f"Combo: {combo.get('name', '')}")
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
            lines.append(f"{idx}. {p.get('name', '')}: {p.get('text', '')}")
    return "\n".join(lines)


def build_profile_context(profile: dict) -> str:
    if not profile:
        return "Chưa có thêm thông tin cụ thể về tuổi, giới tính hay bệnh nền."
    parts: list[str] = []
    if profile.get("age"):
        parts.append(f"Tuổi khoảng: {profile['age']}.")
    if profile.get("gender"):
        parts.append(f"Giới tính: {profile['gender']}.")
    if profile.get("has_chronic") is True:
        parts.append("Có bệnh nền (chi tiết chưa rõ).")
    elif profile.get("has_chronic") is False:
        parts.append("Không có bệnh nền.")
    return " ".join(parts)


# ========= CÂU HỎI LÀM RÕ =========
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
        "Về *thừa cân, béo phì*, anh/chị cho em biết thêm:\n"
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
    "default": (
        "Để em hiểu rõ hơn và tư vấn đúng, anh/chị cho em biết thêm:\n"
        "- Triệu chứng chính anh/chị đang gặp là gì và kéo dài bao lâu rồi?\n"
        "- Anh/chị bao nhiêu tuổi, giới tính gì và có bệnh nền/đang dùng thuốc gì không?\n"
        "- Mục tiêu của anh/chị là giảm triệu chứng, phòng tái phát hay nâng tổng thể sức khoẻ ạ?"
    ),
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
                        "Dữ liệu nội bộ của WELLLAB:\n"
                        + intent_text
                        + "\n\n[HỒ SƠ KHÁCH]: "
                        + profile_ctx
                        + "\n\n[COMBO LIÊN QUAN]:\n"
                        + combo_ctx
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print("Lỗi gọi OpenAI:", e)
        return "Hiện hệ thống AI đang bận, em xin phép anh/chị thử lại sau một chút nhé."


# ========= CÂU CHÀO ĐƠN GIẢN =========
def is_simple_greeting(text: str) -> bool:
    t = text.lower().strip()
    simple = ["chào", "chao", "xin chào", "hi", "hello", "alo", "chao em", "chào em"]
    return any(t == s or t.startswith(s + " ") for s in simple)


def greeting_reply_short() -> str:
    return (
        "Em chào anh/chị 👋\n"
        "Anh/chị cứ tiếp tục chia sẻ nhu cầu hoặc câu hỏi của mình, em luôn sẵn sàng lắng nghe ạ. 😊"
    )


# ========= GỬI TIN =========
def send_message(chat_id: int, text: str, keyboard=None):
    try:
        log_event(chat_id, "bot", text, extra={"source": "bot_reply"})
    except Exception as e:
        print("Lỗi log bot:", e)

    payload: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if keyboard:
        payload["reply_markup"] = {
            "keyboard": keyboard,
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    try:
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json=payload,
            timeout=10,
        )
    except Exception as e:
        print("Lỗi gửi message về Telegram:", e)


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

    tg_user = message.get("from") or {}
    user_id = tg_user.get("id", chat_id)
    profile = get_or_create_user_profile(user_id, tg_user)

    log_event(
        user_id,
        "user",
        text_stripped,
        extra={"username": profile.get("username"), "name": profile.get("name")},
    )

    session = get_session(chat_id)

    # ----- LỆNH CƠ BẢN -----
    if text_stripped.startswith("/start"):
        session["mode"] = "customer"
        session["intent"] = None
        session["profile"] = {}
        session["stage"] = "await_need"
        session["first_issue"] = None
        session["need"] = None
        session["last_combo"] = None

        send_message(
            chat_id,
            build_welcome_message(),
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    if text_stripped.lower() == "/tvv":
        session["mode"] = "tvv"
        send_message(
            chat_id,
            "Đã chuyển sang *chế độ TƯ VẤN VIÊN*. Anh/chị có thể hỏi về combo, sản phẩm hoặc cách tư vấn cho khách.",
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    if text_stripped.lower() == "/kh":
        session["mode"] = "customer"
        send_message(
            chat_id,
            "Đã chuyển về *chế độ tư vấn khách hàng*.",
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    # ----- MENU NHANH -----
    if "Tư vấn theo triệu chứng" in text_stripped:
        session["need"] = "health"
        session["stage"] = "start"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Dạ, anh/chị giúp em mô tả *triệu chứng hoặc vấn đề sức khỏe chính* mình đang gặp ạ:\n"
            "- Đau/khó chịu ở đâu? kéo dài bao lâu rồi?\n"
            "- Anh/chị bao nhiêu tuổi, có bệnh nền/đang dùng thuốc gì không?\n"
            "- Mục tiêu là giảm triệu chứng, phòng tái phát hay nâng sức khỏe tổng thể?"
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="health", intent=None)
        return "ok", 200

    if "Tư vấn theo combo / sản phẩm" in text_stripped:
        session["need"] = "product"
        session["stage"] = "product_clarify"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Dạ, để em tư vấn ĐÚNG sản phẩm nhất, anh/chị cho em biết thêm một chút ạ:\n"
            "- Anh/chị đang muốn cải thiện vấn đề sức khỏe nào (ví dụ: ngủ kém, đau dạ dày, gan yếu...)?\n"
            "- Anh/chị đã có combo/sản phẩm nào của WELLLAB trong tay chưa hay đang tìm hiểu từ đầu?\n"
            "Anh/chị có thể gửi *tên combo, mã combo* (nếu có) hoặc mô tả mục tiêu chính, em sẽ gợi ý thật phù hợp ạ."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="product", intent=None)
        return "ok", 200

    if "Hỏi chính sách mua hàng" in text_stripped:
        session["need"] = "policy"
        session["stage"] = "start"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Dạ, anh/chị muốn hỏi rõ hơn về *mua hàng, giao hàng hay thanh toán* ạ?\n"
            "Anh/chị cứ hỏi cụ thể: ví dụ *phí ship*, *thời gian giao*, *hình thức thanh toán*, *đổi trả*..."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="policy", intent=None)
        return "ok", 200

    # ----- CHÀO HỎI -----
    if is_simple_greeting(text_stripped):
        if not session.get("need"):
            session["stage"] = "await_need"
            send_message(
                chat_id,
                build_welcome_message(),
                keyboard=get_main_menu_keyboard(),
            )
        else:
            send_message(chat_id, greeting_reply_short())
        return "ok", 200

    # ----- NÓI “KHÔNG CÓ VẤN ĐỀ SỨC KHOẺ” -----
    if is_no_health_intent(text_stripped):
        session["need"] = "other"
        session["intent"] = None
        session["stage"] = "start"
        session["first_issue"] = None

        reply = (
            "Dạ vâng anh/chị 😊\n"
            "Nếu hiện tại anh/chị *không có vấn đề sức khỏe cụ thể*, em vẫn có thể hỗ trợ:\n"
            "- Gợi ý các combo/sản phẩm chăm sóc sức khỏe tổng thể, phòng ngừa.\n"
            "- Giải đáp thắc mắc về thành phần, cách dùng, liệu trình WELLLAB.\n"
            "- Thông tin về chính sách mua hàng, giao hàng, thanh toán.\n\n"
            "Anh/chị muốn *tìm hiểu sản phẩm*, *xây dựng liệu trình dự phòng* hay *hỏi về chính sách* ạ?"
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need="other", intent=None)
        return "ok", 200

    # ----- CẬP NHẬT HỒ SƠ CƠ BẢN -----
    prof_update = extract_profile(text_stripped)
    if prof_update:
        session["profile"] = {**session.get("profile", {}), **prof_update}

    # ----- FAQ / OBJECTION (KHÔNG TỐN TOKEN) -----
    faq_answer = try_answer_faq(text_stripped)
    if faq_answer:
        send_message(chat_id, faq_answer)
        need_auto = session.get("need") or detect_need(text_stripped)
        session["need"] = need_auto
        touch_user_stats(profile, need=need_auto, intent=None)
        return "ok", 200

    obj_answer = try_answer_objection(text_stripped)
    if obj_answer:
        send_message(chat_id, obj_answer)
        need_auto = session.get("need") or detect_need(text_stripped)
        session["need"] = need_auto
        touch_user_stats(profile, need=need_auto, intent=None)
        return "ok", 200

    # ====== XÁC ĐỊNH NEED ======
    lower = text_stripped.lower()
    explicit_need = None

    if any(kw in lower for kw in ["sản phẩm", "san pham", "combo", "liệu trình", "lieu trinh"]):
        explicit_need = "product"

    if any(
        kw in lower
        for kw in [
            "chính sách",
            "mua hàng",
            "dat hang",
            "đặt hàng",
            "ship",
            "giao hàng",
            "thanh toán",
            "thanh toan",
            "đổi trả",
            "doi tra",
            "bảo hành",
            "bao hanh",
        ]
    ):
        explicit_need = "policy"

    if any(
        kw in lower
        for kw in [
            "sức khỏe",
            "suc khoe",
            "đau ",
            "bị đau",
            "benh",
            "bệnh",
            "triệu chứng",
            "huyết áp",
            "tieu duong",
            "tiểu đường",
            "mỡ máu",
            "gan",
            "thận",
            "da cơ địa",
            "vảy nến",
            "mat ngu",
            "mất ngủ",
            "ho",
            "khó thở",
            "kho tho",
            "viem",
        ]
    ):
        explicit_need = explicit_need or "health"

    if explicit_need:
        session["need"] = explicit_need
        if session.get("stage") == "await_need":
            session["stage"] = "start"
    elif not session.get("need") or session.get("stage") == "await_need":
        session["need"] = detect_need(text_stripped)
        session["stage"] = "start"

    need = session.get("need") or "other"

    # ====== NHÁNH CHÍNH SÁCH ======
    if need == "policy":
        faq_answer = try_answer_faq(text_stripped)
        if faq_answer:
            send_message(chat_id, faq_answer)
            touch_user_stats(profile, need=need, intent=None)
            return "ok", 200

        combo = None
        reply = call_openai_for_answer(
            "Khách đang hỏi về CHÍNH SÁCH hoặc MUA HÀNG. "
            "Hãy trả lời ngắn gọn, rõ ràng, thân thiện. Không tư vấn bệnh hoặc liệu trình.\n\n"
            "Câu hỏi của khách: " + text_stripped,
            session,
            combo,
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== NHÁNH SẢN PHẨM / COMBO ======
    if need == "product":
        last_combo = session.get("last_combo")

        # 1. Nếu khách hỏi link/website của "các sản phẩm này" và đã có combo gần nhất
        if last_combo and any(
            kw in lower for kw in ["link", "đường link", "duong link", "url", "website", "trang web"]
        ):
            lines: list[str] = [
                f"Dạ, link các sản phẩm trong *{last_combo.get('name', '')}* đây ạ:"
            ]
            products = last_combo.get("products", [])
            for idx, p in enumerate(products, start=1):
                name = p.get("name", "")
                code = p.get("code", "")
                url_p = p.get("url", "")
                line = f"{idx}. {name}"
                if code:
                    line += f" ({code})"
                if url_p:
                    line += f": {url_p}"
                lines.append(line)
            lines.append(
                "\nAnh/chị cần em giải thích thêm về thành phần hoặc cách dùng của sản phẩm nào trong combo này không ạ?"
            )
            send_message(chat_id, "\n".join(lines))
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 2. Khách gõ tên/mã combo cụ thể
        matches = search_combo_by_text(text_stripped, top_k=1)
        if matches:
            combo = matches[0]
            session["last_combo"] = combo
            if not session.get("intent"):
                session["intent"] = "product_info"

            reply = call_openai_for_answer(
                (
                    "Khách đang hỏi về một COMBO/SẢN PHẨM cụ thể trong danh mục WELLLAB.\n"
                    "Hãy giải thích rõ ràng, dễ hiểu cho khách về combo này, dựa trên dữ liệu nội bộ.\n"
                    "- Không bịa thêm combo mới.\n"
                    "- Nhấn mạnh: sản phẩm chỉ hỗ trợ sức khoẻ, không phải thuốc chữa bệnh.\n"
                    "- Thực hiện tư vấn có TÂM theo quy trình 5 bước trong system prompt.\n\n"
                    f"Câu hỏi gốc của khách: {text_stripped}"
                ),
                session,
                combo,
            )
            send_message(chat_id, reply)
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 3. Không nhận diện được combo -> hỏi rõ thêm
        session["stage"] = "product_clarify"
        ask = (
            "Để em tư vấn đúng sản phẩm nhất cho anh/chị, em cần hiểu rõ hơn một chút ạ:\n"
            "- Anh/chị đang muốn cải thiện vấn đề gì (ví dụ: huyết áp, tiểu đường, gan, tiêu hoá, da, giấc ngủ...)?\n"
            "- Anh/chị có đang dùng thuốc hoặc sản phẩm hỗ trợ nào khác không?\n"
            "Sau khi biết rõ tình trạng và mục tiêu, em sẽ gợi ý combo phù hợp nhất, tránh thừa/thiếu sản phẩm cho anh/chị ạ."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== OTHER (CHƯA RÕ) ======
    if need == "other" and not detect_intent_from_text(text_stripped):
        reply = (
            "Để em hỗ trợ đúng hơn, anh/chị cho em biết thêm một chút ạ:\n"
            "- Anh/chị đang muốn *tìm giải pháp cho vấn đề sức khỏe*, *tìm hiểu sản phẩm* hay *hỏi về chính sách mua hàng*?\n"
            "- Nếu có triệu chứng hoặc mục tiêu sức khỏe cụ thể (ví dụ: mất ngủ, viêm da, huyết áp...), anh/chị mô tả giúp em nhé."
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== FLOW SỨC KHOẺ ======
    if need == "health":
        new_intent = detect_intent_from_text(text_stripped)
        if new_intent:
            session["intent"] = new_intent

        intent = session.get("intent")
        stage = session.get("stage", "start")

        touch_user_stats(profile, need=need, intent=intent)

        # 1. ĐANG CLARIFY -> coi đây là thông tin bổ sung, tư vấn luôn
        if stage == "clarify":
            issue = session.get("first_issue") or ""
            if not issue:
                session["first_issue"] = text_stripped
                issue = text_stripped

            combined_user_text = (
                "Mô tả ban đầu của khách: " + issue + "\n\n"
                "Thông tin bổ sung khách vừa cung cấp: " + text_stripped
            )

            combo = choose_combo(intent)
            session["last_combo"] = combo
            session["stage"] = "advise"
            reply = call_openai_for_answer(combined_user_text, session, combo)
            send_message(chat_id, reply)
            return "ok", 200

        # 2. CHƯA CÓ INTENT RÕ
        if not intent:
            question = get_clarify_question(None)
            session["stage"] = "clarify"
            if not session.get("first_issue"):
                session["first_issue"] = text_stripped
            send_message(chat_id, question)
            return "ok", 200

        # 3. CÓ INTENT, ĐANG Ở START
        if stage in ("start", None):
            session["first_issue"] = text_stripped
            session["stage"] = "clarify"
            question = get_clarify_question(intent)
            send_message(chat_id, question)
            return "ok", 200

        # 4. GIAI ĐOẠN ADVISE -> câu hỏi bổ sung
        if stage == "advise":
            combo = choose_combo(intent)
            session["last_combo"] = combo
            reply = call_openai_for_answer(text_stripped, session, combo)
            send_message(chat_id, reply)
            return "ok", 200

        # Fallback trong health
        combo = choose_combo(intent)
        session["last_combo"] = combo
        reply = call_openai_for_answer(text_stripped, session, combo)
        send_message(chat_id, reply)
        return "ok", 200

    # ====== FALLBACK CHUNG ======
    intent = session.get("intent")
    combo = choose_combo(intent)
    session["last_combo"] = combo
    reply = call_openai_for_answer(text_stripped, session, combo)
    send_message(chat_id, reply)
    touch_user_stats(profile, need=need, intent=intent)
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
