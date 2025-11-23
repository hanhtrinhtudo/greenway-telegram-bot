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
PRODUCTS_PATH = DATA_DIR / "welllab_products.json"    # danh mục sản phẩm lẻ

WELLLAB_CATALOG = load_json(CATALOG_PATH, [])
SYMPTOM_RULES = load_json(SYMPTOMS_PATH, [])
FAQ_LIST = load_json(FAQ_PATH, [])
OBJECTION_LIST = load_json(OBJECTIONS_PATH, [])
WELLLAB_PRODUCTS = load_json(PRODUCTS_PATH, [])


# ========= TIỆN ÍCH CHUẨN HÓA =========
def normalize_text(s: str) -> str:
    """Bỏ dấu, về thường để so khớp tên linh hoạt hơn."""
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


def search_product_by_text(query: str, top_k: int = 1) -> list[dict]:
    """
    Tìm sản phẩm lẻ trong welllab_products.json theo name/code.
    Ưu tiên: trùng mã code, sau đó đến tên.
    """
    q = normalize_text(query)
    if not q or not WELLLAB_PRODUCTS:
        return []

    results: list[tuple[int, dict]] = []

    for p in WELLLAB_PRODUCTS:
        name = normalize_text(p.get("name", ""))
        code = normalize_text(p.get("code", ""))
        haystack = f"{name} {code}"

        score = 0
        # nếu khớp mã code nguyên chuỗi -> cộng điểm cao
        if code and code in q:
            score += 5
        for token in q.split():
            if token and token in haystack:
                score += 1

        if score > 0:
            results.append((score, p))

    results.sort(key=lambda x: x[0], reverse=True)
    return [prod for score, prod in results[:top_k]]


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
    # lưu ngay mỗi lần cập nhật
    save_users_store(USERS_STORE)
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
            "stage": "await_need",  # await_need -> start -> clarify -> advise
            "first_issue": None,
            "need": None,
            "last_combo": None,          # lưu combo đã tư vấn gần nhất
            "clarify_rounds": 0,         # số vòng hỏi rõ
            "issue_summary": "",         # tóm tắt vấn đề sức khỏe
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
    "PHONG CÁCH TƯ VẤN (PHA CLARIFY):\n"
    "- Khi pha = clarify: chỉ đặt tối đa 2–3 câu hỏi NGẮN để làm rõ thông tin còn thiếu.\n"
    "- Không được lặp lại các câu hỏi khách đã trả lời, không gợi ý combo, không chốt bán hàng.\n\n"
    "PHONG CÁCH TƯ VẤN (PHA ADVISE hoặc FOLLOW_UP):\n"
    "- Coi như đã có đủ thông tin cơ bản, KHÔNG hỏi lại những dữ liệu nền (thời gian bị, tuổi, bệnh nền...) trừ khi khách chưa nói.\n"
    "- Tóm tắt ngắn gọn lại vấn đề của khách, sau đó gợi ý hướng xử lý và combo/sản phẩm phù hợp.\n"
    "- Mỗi lần trả lời tối đa khoảng 8–10 dòng chat, ưu tiên gạch đầu dòng.\n"
    "- Luôn kết thúc bằng một câu hỏi mở rất ngắn (ví dụ: 'Anh/chị thấy như vậy ổn không ạ?' hoặc 'Anh/chị cần em giải thích thêm phần nào không?').\n"
)

TVV_SYSTEM_EXTRA = (
    "Ngữ cảnh: Người đang trao đổi với bạn là *TƯ VẤN VIÊN* của công ty, không phải khách hàng cuối.\n"
    "- Trả lời như đang huấn luyện nội bộ: giải thích mục tiêu từng combo, cách đặt câu hỏi, cách xử lý thắc mắc.\n"
    "- Luôn nhắc lại quy trình tư vấn có TÂM để tư vấn viên áp dụng với khách.\n"
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
        "hoặc *“Anh muốn hỏi về combo/sản phẩm…”* để em hỗ trợ ạ. 💚"
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
        "giá bao nhiêu", "bao nhiêu tiền", "mã", "code",
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


def build_product_context(prod: dict | None) -> str:
    if not prod:
        return "Hiện chưa xác định được sản phẩm cụ thể."
    lines: list[str] = []
    lines.append(f"Sản phẩm: {prod.get('name', '')}")
    code = prod.get("code")
    if code:
        lines.append(f"Mã: {code}")
    price = prod.get("price")
    if price:
        lines.append(f"Giá: {price}")
    ingredients = prod.get("ingredients")
    if ingredients:
        lines.append(f"Thành phần: {ingredients}")
    usage = prod.get("usage")
    if usage:
        lines.append(f"Cách dùng: {usage}")
    benefits = prod.get("benefits")
    if benefits:
        lines.append(f"Công dụng: {benefits}")
    link = prod.get("link")
    if link:
        lines.append(f"Link: {link}")
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
def call_openai_for_answer(
    user_text: str,
    session: dict,
    combo: dict | None = None,
    product: dict | None = None,
    phase: str = "advise",
) -> str:
    """
    phase: 'clarify', 'advise', 'follow_up', 'policy', 'product_info'...
    """
    mode = session.get("mode", "customer")
    intent = session.get("intent")
    profile = session.get("profile", {})
    issue_summary = session.get("issue_summary") or session.get("first_issue") or ""

    sys_prompt = BASE_SYSTEM_PROMPT
    if mode == "tvv":
        sys_prompt += "\n" + TVV_SYSTEM_EXTRA

    # Xây context combo / product
    if product:
        item_ctx = build_product_context(product)
    else:
        item_ctx = build_combo_context(combo)

    profile_ctx = build_profile_context(profile)
    intent_text = f"Intent hiện tại: {intent or 'chưa rõ'}."
    phase_text = f"Pha hội thoại hiện tại: {phase}."
    issue_text = issue_summary or "Chưa có tóm tắt chi tiết."

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
                        + "\n"
                        + phase_text
                        + "\n\n[HỒ SƠ KHÁCH]: "
                        + profile_ctx
                        + "\n\n[TÓM TẮT VẤN ĐỀ]: "
                        + issue_text
                        + "\n\n[SẢN PHẨM/COMBO LIÊN QUAN]:\n"
                        + item_ctx
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
        session.clear()
        session.update(
            {
                "mode": "customer",
                "intent": None,
                "profile": {},
                "stage": "await_need",
                "first_issue": None,
                "need": None,
                "last_combo": None,
                "clarify_rounds": 0,
                "issue_summary": "",
            }
        )

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
        session["clarify_rounds"] = 0
        session["issue_summary"] = ""
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
        session["clarify_rounds"] = 0
        session["issue_summary"] = ""
        ask = (
            "Dạ, để em tư vấn ĐÚNG sản phẩm nhất, anh/chị cho em biết thêm một chút ạ:\n"
            "- Anh/chị đang muốn cải thiện vấn đề sức khỏe nào (ví dụ: ngủ kém, đau dạ dày, gan yếu...)?\n"
            "- Anh/chị đã có combo/sản phẩm nào của WELLLAB trong tay chưa hay đang tìm hiểu từ đầu?\n"
            "Anh/chị có thể gửi *tên sản phẩm, mã sản phẩm hoặc tên combo*, em sẽ gợi ý thật phù hợp ạ."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="product", intent=None)
        return "ok", 200

    if "Hỏi chính sách mua hàng" in text_stripped:
        session["need"] = "policy"
        session["stage"] = "start"
        session["intent"] = None
        session["first_issue"] = None
        session["clarify_rounds"] = 0
        session["issue_summary"] = ""
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
        session["clarify_rounds"] = 0
        session["issue_summary"] = ""

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

    if any(kw in lower for kw in ["sản phẩm", "san pham", "liệu trình", "lieu trinh", "mã", "code"]):
        explicit_need = "product"

    if any(
        kw in lower
        for kw in [
            "combo", "bộ sản phẩm", "bo san pham",
        ]
    ):
        explicit_need = explicit_need or "product"

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
        if not explicit_need:
            explicit_need = "health"

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

        reply = call_openai_for_answer(
            "Khách đang hỏi về CHÍNH SÁCH hoặc MUA HÀNG. "
            "Hãy trả lời ngắn gọn, rõ ràng, thân thiện. Không tư vấn bệnh hoặc liệu trình.\n\n"
            "Câu hỏi của khách: " + text_stripped,
            session,
            combo=None,
            product=None,
            phase="policy",
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== NHÁNH SẢN PHẨM / COMBO ======
    if need == "product":
        # Nếu câu này thực chất mô tả triệu chứng -> chuyển sang health
        if detect_need(text_stripped) == "health":
            session["need"] = "health"
            session.setdefault("stage", "start")
            need = "health"
        else:
            # 1. Thử nhận diện sản phẩm lẻ trước
            prod_matches = search_product_by_text(text_stripped, top_k=1)
            if prod_matches:
                product = prod_matches[0]
                reply = call_openai_for_answer(
                    (
                        "Khách đang hỏi về MỘT SẢN PHẨM CỤ THỂ trong danh mục WELLLAB.\n"
                        "Hãy giải thích rõ ràng, dễ hiểu về sản phẩm này dựa trên dữ liệu nội bộ.\n"
                        "- Không bịa thêm sản phẩm.\n"
                        "- Nhấn mạnh: sản phẩm chỉ hỗ trợ sức khoẻ, không phải thuốc chữa bệnh.\n"
                        "- Tập trung trả lời đúng câu hỏi của khách, không lan man sang vấn đề khác.\n\n"
                        f"Câu hỏi/nhu cầu của khách: {text_stripped}"
                    ),
                    session,
                    combo=None,
                    product=product,
                    phase="product_info",
                )
                send_message(chat_id, reply)
                touch_user_stats(profile, need="product", intent="product_info")
                return "ok", 200

            # 2. Nếu khách hỏi rõ về combo -> tìm combo
            if "combo" in lower or "bộ sản phẩm" in lower or "bo san pham" in lower:
                matches = search_combo_by_text(text_stripped, top_k=1)
                if matches:
                    combo = matches[0]
                    session["last_combo"] = combo
                    if not session.get("intent"):
                        session["intent"] = "product_info"

                    reply = call_openai_for_answer(
                        (
                            "Khách đang hỏi về một COMBO CỤ THỂ trong danh mục WELLLAB.\n"
                            "Hãy giải thích rõ ràng, dễ hiểu cho khách về combo này, dựa trên dữ liệu nội bộ.\n"
                            "- Không bịa thêm combo mới.\n"
                            "- Nhấn mạnh: sản phẩm chỉ hỗ trợ sức khoẻ, không phải thuốc chữa bệnh.\n"
                            "- Trả lời đúng trọng tâm câu hỏi.\n\n"
                            f"Câu hỏi gốc của khách: {text_stripped}"
                        ),
                        session,
                        combo=combo,
                        product=None,
                        phase="product_info",
                    )
                    send_message(chat_id, reply)
                    touch_user_stats(profile, need="product", intent="product_info")
                    return "ok", 200

            # 3. Không nhận diện được gì -> hỏi thêm
            session["stage"] = "product_clarify"
            ask = (
                "Để em tư vấn đúng sản phẩm nhất cho anh/chị, em cần hiểu rõ hơn một chút ạ:\n"
                "- Anh/chị đang muốn cải thiện vấn đề gì (ví dụ: huyết áp, tiểu đường, gan, tiêu hoá, da, giấc ngủ...)?\n"
                "- Anh/chị có đang dùng thuốc hoặc sản phẩm hỗ trợ nào khác không?\n"
                "Sau khi biết rõ tình trạng và mục tiêu, em sẽ gợi ý sản phẩm/combo phù hợp nhất, tránh thừa/thiếu cho anh/chị ạ."
            )
            send_message(chat_id, ask)
            touch_user_stats(profile, need="product", intent=None)
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
        # Giữ intent nếu đã có, tránh nhảy lung tung
        new_intent = detect_intent_from_text(text_stripped)
        if new_intent and not session.get("intent"):
            session["intent"] = new_intent

        intent = session.get("intent")
        stage = session.get("stage", "start")

        touch_user_stats(profile, need=need, intent=intent)

        # START: lần đầu mô tả vấn đề
        if stage in ("start", None):
            session["first_issue"] = text_stripped
            session["issue_summary"] = text_stripped
            session["clarify_rounds"] = 0
            session["stage"] = "clarify"
            question = get_clarify_question(intent)
            send_message(chat_id, question)
            return "ok", 200

        # CLARIFY: khách đang trả lời câu hỏi làm rõ
        if stage == "clarify":
            issue = session.get("issue_summary") or session.get("first_issue") or ""
            if issue:
                issue = issue + " | Thông tin bổ sung: " + text_stripped
            else:
                issue = text_stripped
            session["issue_summary"] = issue
            session["clarify_rounds"] = int(session.get("clarify_rounds") or 0) + 1

            # Sau 1–2 vòng clarify thì chuyển sang ADVISE
            if session["clarify_rounds"] >= 1:
                combo = choose_combo(intent)
                session["last_combo"] = combo
                session["stage"] = "advise"
                reply = call_openai_for_answer(
                    "Tóm tắt toàn bộ vấn đề sức khỏe khách đang gặp:\n" + issue,
                    session,
                    combo=combo,
                    product=None,
                    phase="advise",
                )
                send_message(chat_id, reply)
                return "ok", 200

            # Nếu vẫn muốn hỏi thêm (trường hợp hiếm)
            question = get_clarify_question(intent)
            send_message(chat_id, question)
            return "ok", 200

        # ADVISE: khách hỏi thêm sau khi đã được tư vấn combo
        if stage == "advise":
            combo = choose_combo(intent)
            session["last_combo"] = combo
            reply = call_openai_for_answer(
                text_stripped,
                session,
                combo=combo,
                product=None,
                phase="follow_up",
            )
            send_message(chat_id, reply)
            return "ok", 200

        # Fallback trong health
        combo = choose_combo(intent)
        session["last_combo"] = combo
        reply = call_openai_for_answer(
            text_stripped,
            session,
            combo=combo,
            product=None,
            phase="advise",
        )
        send_message(chat_id, reply)
        return "ok", 200

    # ====== FALLBACK CHUNG ======
    intent = session.get("intent")
    combo = choose_combo(intent)
    session["last_combo"] = combo
    reply = call_openai_for_answer(
        text_stripped,
        session,
        combo=combo,
        product=None,
        phase="advise",
    )
    send_message(chat_id, reply)
    touch_user_stats(profile, need=need, intent=intent)
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
