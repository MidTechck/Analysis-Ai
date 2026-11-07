from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
import imaplib
import email
import email.utils
from email.header import decode_header
import re
import csv
import io
import json
import requests
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta, timezone, date
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-set-SECRET_KEY-in-env")

# --- MULTI-PROVIDER LLM ENGINE (silent failover, auto model discovery) ---
PROVIDERS = []

def discover_providers():
    """Discover available free models across providers that have keys set. Fast timeouts so startup stays quick."""
    global PROVIDERS
    PROVIDERS = []

    # Gemini (Google)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_actual_gemini_api_key_here":
        try:
            res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}", timeout=4)
            if res.ok:
                models = [m['name'] for m in res.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                preferred = ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]
                chosen = next((p for p in preferred if p in models), models[0] if models else None)
                if chosen:
                    PROVIDERS.append({
                        "name": "gemini",
                        "model": chosen,
                        "key": gemini_key,
                        "type": "gemini"
                    })
                    print(f"[LLM] Gemini ready: {chosen}")
        except Exception as e:
            print(f"[LLM] Gemini discovery skipped: {e}")

    # Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            res = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {groq_key}"}, timeout=4)
            if res.ok:
                models = [m['id'] for m in res.json().get('data', [])]
                preferred = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b"]
                chosen = next((p for p in preferred if p in models), models[0] if models else "llama-3.1-8b-instant")
                PROVIDERS.append({
                    "name": "groq",
                    "model": chosen,
                    "key": groq_key,
                    "type": "openai",
                    "base": "https://api.groq.com/openai/v1"
                })
                print(f"[LLM] Groq ready: {chosen}")
        except Exception as e:
            print(f"[LLM] Groq discovery skipped: {e}")

    # NVIDIA NIM
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        PROVIDERS.append({
            "name": "nvidia",
            "model": "meta/llama-3.1-8b-instruct",
            "key": nvidia_key,
            "type": "openai",
            "base": "https://integrate.api.nvidia.com/v1"
        })
        print("[LLM] NVIDIA NIM ready (default model)")

    # xAI / Grok
    xai_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if xai_key:
        try:
            res = requests.get("https://api.x.ai/v1/models", headers={"Authorization": f"Bearer {xai_key}"}, timeout=4)
            if res.ok:
                models = [m['id'] for m in res.json().get('data', [])]
                preferred = ["grok-4.7", "grok-3", "grok-2-latest", "grok-beta"]
                chosen = next((p for p in preferred if any(p in m for m in models)), models[0] if models else "grok-beta")
                PROVIDERS.append({
                    "name": "xai",
                    "model": chosen,
                    "key": xai_key,
                    "type": "openai",
                    "base": "https://api.x.ai/v1"
                })
                print(f"[LLM] xAI/Grok ready: {chosen}")
        except Exception as e:
            print(f"[LLM] xAI discovery skipped: {e}")

    if not PROVIDERS:
        print("[LLM WARNING] No working providers found. Set at least one API key in .env")
    else:
        print(f"[LLM] {len(PROVIDERS)} provider(s) available for silent failover")

def query_llm(prompt, max_retries=2):
    """Call any available provider. Silently fail over on quota/errors."""
    if not PROVIDERS:
        raise Exception("No LLM providers configured")

    last_error = None
    for provider in PROVIDERS:
        for attempt in range(max_retries):
            try:
                if provider["type"] == "gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/{provider['model']}:generateContent?key={provider['key']}"
                    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
                    data = res.json()
                    if "error" in data:
                        raise Exception(data["error"].get("message", "Gemini error"))
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

                else:  # OpenAI-compatible (Groq, NVIDIA, xAI)
                    url = f"{provider['base']}/chat/completions"
                    headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json"}
                    payload = {
                        "model": provider["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1024
                    }
                    res = requests.post(url, headers=headers, json=payload, timeout=30)
                    data = res.json()
                    if "error" in data:
                        raise Exception(data["error"].get("message", "Provider error"))
                    return data["choices"][0]["message"]["content"].strip()

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(x in err_str for x in ["quota", "rate", "limit", "429", "billing"]):
                    print(f"[LLM] {provider['name']} rate-limited, trying next...")
                    break  # skip remaining retries for this provider
                time.sleep(0.5)
                continue
        # If we exhausted retries for this provider, try next
    raise Exception(f"All providers failed. Last error: {last_error}")

def parse_llm_json(raw_text):
    """Clean and parse JSON from any LLM."""
    cleaned = raw_text.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Model added extra words around the JSON - pull out the {...} block
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            return json.loads(m.group(0))
        raise

# --- STARTUP ---
print("\n" + "="*50)
print("SYSTEM BOOT DIAGNOSTICS & MULTI-PROVIDER LLM:")
discover_providers()
print("="*50 + "\n")

# --- USER STORE (username isolates all data) ---
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def get_memory_file(username):
    safe = username.replace('@', '_').replace('.', '_').replace(' ', '_')
    return f"memory_{safe}.json"

def load_memory(username):
    file = get_memory_file(username)
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(username, history):
    file = get_memory_file(username)
    with open(file, 'w') as f:
        json.dump(history, f)

# --- IMAP HELPERS ---
@contextmanager
def imap_connection(email_acc, password, select_inbox=True):
    mail = imaplib.IMAP4_SSL("mail.privateemail.com")
    try:
        mail.login(email_acc, password)
        if select_inbox:
            mail.select("inbox")
        yield mail
    finally:
        try:
            mail.logout()
        except:
            pass

def build_imap_query(task_type, target_subject, date_filter):
    target_subject = (target_subject or "").replace('"', '')
    date_filter = clean_date_filter(date_filter)
    if task_type == "client_tally":
        return f'(ALL {date_filter})' if date_filter != "NONE" else 'ALL'
    else:
        if date_filter != "NONE":
            return f'(SUBJECT "{target_subject}" {date_filter})'
        return f'SUBJECT "{target_subject}"'

def execute_imap_search(email_acc, password, task_type, target_subject, date_filter, limit=None):
    records = []
    match_count = 0
    total_inbox = 0
    try:
        with imap_connection(email_acc, password) as mail:
            _, all_msgs = mail.search(None, 'ALL')
            total_inbox = len(all_msgs[0].split()) if all_msgs[0] else 0

            imap_query = build_imap_query(task_type, target_subject, date_filter)
            _, filtered = mail.search(None, imap_query)
            matching_ids = filtered[0].split() if filtered[0] else []
            match_count = len(matching_ids)
            if limit is None:
                limit = len(matching_ids)

            if task_type == "client_tally":
                unique_clients = set()
                for e_id in reversed(matching_ids[-limit:]):
                    _, msg_data = mail.fetch(e_id, '(BODY.PEEK[HEADER.FIELDS (TO)])')
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            raw_to = msg.get("To", "")
                            if raw_to:
                                _, clean_to_email = email.utils.parseaddr(raw_to)
                                if clean_to_email and "@" in clean_to_email:
                                    unique_clients.add(clean_to_email.lower())
                match_count = len(unique_clients)
                for client in sorted(unique_clients):
                    records.append({"client_email": client})
            else:
                # Original full RFC822 fetch logic (restored so manual searches return full results)
                for e_id in reversed(matching_ids[-limit:]):
                    _, msg_data = mail.fetch(e_id, '(RFC822)')
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            subject_hdr = decode_header(msg.get("Subject", ""))[0]
                            subject = subject_hdr[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(subject_hdr[1] if subject_hdr[1] else 'utf-8', errors='ignore')
                            raw_to = msg.get("To", "")
                            _, clean_to_email = email.utils.parseaddr(raw_to)
                            raw_date = msg.get("Date", "")
                            try:
                                parsed_date = email.utils.parsedate_to_datetime(raw_date)
                                clean_date = parsed_date.astimezone(LOCAL_TZ).strftime("%d %b %Y, %H:%M")
                            except Exception:
                                clean_date = raw_date
                            
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() in ["text/plain", "text/html"]:
                                        try: body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                        except: pass
                            else:
                                try: body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                                except: pass

                            account_match = re.search(r'(ACC-DF-[\d-]+|INV-DF-[A-Z\d-]+)', body + " " + subject)
                            account_no = account_match.group(1) if account_match else "Nil"
                            records.append({
                                "client_email": clean_to_email,
                                "subject": subject,
                                "date": clean_date,
                                "account_no": account_no
                            })
        return {"success": True, "total_inbox": total_inbox, "match_count": match_count, "records": records}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- DATES (Zambia = CAT, UTC+2, no daylight saving) ---
LOCAL_TZ = timezone(timedelta(hours=2))
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(MONTH_ABBR, 1)}
MONTH_LOOKUP["sept"] = 9

def today_local():
    return datetime.now(LOCAL_TZ).date()

def imap_date(d):
    return f"{d.day:02d}-{MONTH_ABBR[d.month - 1]}-{d.year}"

DATE_FILTER_RE = re.compile(r"^(?:(?:SINCE|BEFORE|ON) \d{1,2}-[A-Za-z]{3}-\d{4}\s*)+$")

def clean_date_filter(date_filter):
    """Only allow plain IMAP date keywords. Anything else becomes NONE."""
    df = (date_filter or "NONE").strip()
    return df if DATE_FILTER_RE.match(df) else "NONE"

_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_SEP = r"(?:to|-|–|until|till|and|through)"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YR = r"(?:\s*,?\s*(\d{4}))?"

def _mk_date(day, mon, year, today):
    try:
        month = MONTH_LOOKUP[mon[:4] if mon.startswith("sept") else mon[:3]]
        d = date(int(year) if year else today.year, month, int(day))
        # No year given and it lands far in the future -> they meant last year
        if not year and d > today + timedelta(days=31):
            d = date(today.year - 1, month, int(day))
        return d
    except (ValueError, KeyError):
        return None

def parse_date_range(text, today=None):
    """Turn words like '6 to 7 November' or 'yesterday and today' into (start, end). Returns None if no date found.
    Explicit calendar dates win over words like 'today'."""
    today = today or today_local()
    t = text.lower()

    # "6 november to 7 november"
    m = re.search(rf"\b{_DAY}\s+{_MON}{_YR}\s*{_SEP}\s*{_DAY}\s+{_MON}{_YR}\b", t)
    if m:
        d1 = _mk_date(m.group(1), m.group(2), m.group(3), today)
        d2 = _mk_date(m.group(4), m.group(5), m.group(6), today)
        if d1 and d2:
            return (min(d1, d2), max(d1, d2))
    # "6 to 7 november" / "6-7 nov 2026"
    m = re.search(rf"\b{_DAY}\s*{_SEP}\s*{_DAY}\s+{_MON}{_YR}\b", t)
    if m:
        d1 = _mk_date(m.group(1), m.group(3), m.group(4), today)
        d2 = _mk_date(m.group(2), m.group(3), m.group(4), today)
        if d1 and d2:
            return (min(d1, d2), max(d1, d2))
    # "november 6 to 7" / "nov 6-7"
    m = re.search(rf"\b{_MON}\s+{_DAY}\s*{_SEP}\s*{_DAY}{_YR}\b", t)
    if m:
        d1 = _mk_date(m.group(2), m.group(1), m.group(4), today)
        d2 = _mk_date(m.group(3), m.group(1), m.group(4), today)
        if d1 and d2:
            return (min(d1, d2), max(d1, d2))
    # "7 november" / "november 7"
    m = re.search(rf"\b{_DAY}\s+{_MON}{_YR}\b", t)
    if m:
        d1 = _mk_date(m.group(1), m.group(2), m.group(3), today)
        if d1:
            return (d1, d1)
    m = re.search(rf"\b{_MON}\s+{_DAY}{_YR}\b", t)
    if m:
        d1 = _mk_date(m.group(2), m.group(1), m.group(3), today)
        if d1:
            return (d1, d1)

    # Relative words
    yesterday = today - timedelta(days=1)
    m = re.search(r"\blast\s+(\d{1,3})\s+days?\b", t)
    if m:
        return (today - timedelta(days=int(m.group(1)) - 1), today)
    if "yesterday" in t and "today" in t:
        return (yesterday, today)
    if "yesterday" in t:
        return (yesterday, yesterday)
    if "today" in t:
        return (today, today)
    if "this week" in t:
        return (today - timedelta(days=today.weekday()), today)
    if "last week" in t:
        start = today - timedelta(days=today.weekday() + 7)
        return (start, start + timedelta(days=6))
    if "this month" in t:
        return (today.replace(day=1), today)
    return None

# --- KNOWN EMAIL SUBJECTS (add more here if you like) ---
SUBJECT_PATTERNS = [
    (re.compile(r"\borders?\s+confirmed\b", re.I), "Order confirmed"),
    (re.compile(r"\bpayments?\s+processed\b", re.I), "Payment processed"),
]

def detect_subjects(text):
    found = []
    for pattern, name in SUBJECT_PATTERNS:
        m = pattern.search(text)
        if m:
            found.append((m.start(), name))
    return [name for _, name in sorted(found)]

def count_emails_by_day(email_acc, password, subjects, start, end):
    """Exact per-day counts, no AI involved. Days are in Zambia time (CAT).
    Searches the folders in MAIL_FOLDERS (default INBOX)."""
    folders = [f.strip() for f in os.getenv("MAIL_FOLDERS", "INBOX").split(",") if f.strip()]
    # Search one day wider each side, then filter exactly by the Date header (server dates can be off by timezone)
    since = imap_date(start - timedelta(days=1))
    before = imap_date(end + timedelta(days=2))
    counts = {s: defaultdict(int) for s in subjects}
    try:
        with imap_connection(email_acc, password, select_inbox=False) as mail:
            for subj in subjects:
                seen = set()
                safe = subj.replace('"', '')
                for folder in folders:
                    typ, _ = mail.select(f'"{folder}"', readonly=True)
                    if typ != "OK":
                        continue
                    _, data = mail.search(None, f'(SUBJECT "{safe}" SINCE {since} BEFORE {before})')
                    ids = data[0].split() if data and data[0] else []
                    for i in range(0, len(ids), 100):
                        chunk = ids[i:i + 100]
                        _, msgs = mail.fetch(b",".join(chunk), '(BODY.PEEK[HEADER.FIELDS (DATE MESSAGE-ID)])')
                        for part in msgs:
                            if not isinstance(part, tuple):
                                continue
                            hdr = email.message_from_bytes(part[1])
                            key = (hdr.get("Message-ID") or "").strip() or f"{folder}:{part[0]!r}"
                            if key in seen:
                                continue
                            try:
                                d = email.utils.parsedate_to_datetime(hdr.get("Date", "")).astimezone(LOCAL_TZ).date()
                            except Exception:
                                continue
                            if start <= d <= end:
                                seen.add(key)
                                counts[subj][d] += 1
        return {"success": True, "counts": counts}
    except Exception as e:
        return {"success": False, "error": str(e)}

def format_count_reply(subjects, counts, start, end):
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    totals = {s: sum(counts[s].values()) for s in subjects}
    same_day = start == end
    head = f"{start.strftime('%d %b %Y')}" if same_day else f"{start.strftime('%d %b')} to {end.strftime('%d %b %Y')}"
    lines = [f"Live count from your mailbox, {head} (Zambia time):", ""]

    if len(days) <= 14:
        for d in days:
            parts = " | ".join(f"{s}: {counts[s].get(d, 0)}" for s in subjects)
            mark = ""
            if len(subjects) > 1:
                mark = "  ✅" if len({counts[s].get(d, 0) for s in subjects}) == 1 else "  ❌"
            lines.append(f"{d.strftime('%a %d %b')} - {parts}{mark}")
        lines.append("")

    lines.append("Total - " + " | ".join(f"{s}: {totals[s]}" for s in subjects))

    if len(subjects) > 1:
        if len(set(totals.values())) == 1:
            lines.append(f"Totals match ({next(iter(totals.values()))} each).")
        else:
            hi = max(totals, key=totals.get)
            lo = min(totals, key=totals.get)
            lines.append(f"Totals don't match: {hi} is {totals[hi] - totals[lo]} higher than {lo}.")
    if not any(totals.values()):
        lines.append("")
        lines.append("Nothing found. If you expected emails here, tell me the exact subject wording and I'll search that.")
    lines.append("")
    lines.append("Dashboard is synced to this search.")
    return "\n".join(lines)

def run_count(email_acc, password, subjects, start, end):
    res = count_emails_by_day(email_acc, password, subjects, start, end)
    if not res["success"]:
        return f"I couldn't read the mailbox just now: {res['error']}"
    session['dashboard_query'] = {
        "task_type": "standard",
        "subject": subjects[0],
        "date_filter": f"SINCE {imap_date(start)} BEFORE {imap_date(end + timedelta(days=1))}",
    }
    session.modified = True
    return format_count_reply(subjects, res["counts"], start, end)

def looks_like_data_request(text):
    return bool(re.search(r"\b(orders?|payments?|emails?|mails?|how many|count|total|numbers?|inbox|confirmed|processed)\b", text, re.I))

def build_chat_context(history):
    """Last few messages, minus old error messages so they don't confuse the model."""
    bad = ("Logic Engine Failed", "I ran into a temporary issue")
    clean = [m for m in history[-8:] if not any(str(m.get('text', '')).startswith(b) for b in bad)]
    return "\n".join(f"{m['role']}: {m['text']}" for m in clean)

def process_agentic_chat(user_message, history, email_acc, password):
    today = today_local()
    chat_context = build_chat_context(history)

    # FAST PATH (no AI, exact numbers): known subjects + a date -> count straight from the mailbox
    subjects = detect_subjects(user_message)
    date_range = parse_date_range(user_message, today)
    if subjects and date_range:
        return run_count(email_acc, password, subjects, date_range[0], date_range[1])

    router_prompt = f"""
    Analyze the user's latest message. Output ONLY valid JSON, no other words.

    If the user wants to count emails per day for one or more subjects over a date range, output:
    {{"action": "COUNT", "subjects": ["subject 1", "subject 2"], "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}

    If the user wants to pull/search emails, output:
    {{"action": "SEARCH", "task_type": "standard", "subject": "exact subject phrase or empty", "date_filter": "NONE or IMAP dates like SINCE 06-Nov-2026 BEFORE 08-Nov-2026", "email_prefix": "optional comma-separated starting letters e.g. d,h,c or empty", "specific_email": "optional exact email to look up or empty"}}

    If the user wants to cross-reference two datasets, output:
    {{"action": "COMPARE", "subject_1": "...", "subject_2": "..."}}

    If general chat (greetings, thanks, questions not about emails), output:
    {{"action": "CHAT"}}

    Date rules: Today is {today.strftime('%A %d %B %Y')} ({today.isoformat()}). "yesterday" is {(today - timedelta(days=1)).isoformat()}.
    For IMAP date_filter, BEFORE is exclusive, so to include 7 Nov use BEFORE 08-Nov-2026.
    Do not guess dates from the chat history - use today's date above.

    History:
    {chat_context}
    User latest: {user_message}
    """

    router_failed = False
    try:
        raw_intent = query_llm(router_prompt)
        intent = parse_llm_json(raw_intent)
    except Exception:
        intent = {"action": "CHAT"}
        router_failed = True

    if intent.get("action") == "COUNT":
        try:
            subs = [str(x).strip() for x in intent.get("subjects", []) if str(x).strip()]
            start = date.fromisoformat(intent["start"])
            end = date.fromisoformat(intent["end"])
            if subs:
                return run_count(email_acc, password, subs, min(start, end), max(start, end))
        except Exception:
            pass
        return "I couldn't work out the subjects or dates. Try: 'count order confirmed and payment processed from 6 to 7 November'."

    if intent.get("action") == "CHAT" and router_failed and looks_like_data_request(user_message):
        return "I couldn't process that just now, so I haven't searched anything. Please send it again in a moment."

    data_context = ""
    if intent.get("action") == "SEARCH":
        prefix = (intent.get('email_prefix') or '').strip().lower()
        specific = (intent.get('specific_email') or '').strip().lower()
        # Only fetch every email when a filter needs them; otherwise fetch 25 and use the true total
        fetch_limit = None if (prefix or specific) else 25
        results = execute_imap_search(email_acc, password, intent.get('task_type', 'standard'), intent.get('subject', ''), clean_date_filter(intent.get('date_filter', 'NONE')), limit=fetch_limit)
        if results['success']:
            records = results.get('records', [])
            # Apply optional filters
            if specific:
                records = [r for r in records if r.get('client_email', '').lower() == specific]
            elif prefix:
                prefixes = [p.strip() for p in prefix.split(',') if p.strip()]
                records = [r for r in records if any(r.get('client_email', '').lower().startswith(p) for p in prefixes)]
            
            count = len(records) if (prefix or specific) else results['match_count']
            # Inject real records (limit to avoid huge prompts)
            sample = records[:25]
            sample_text = "\n".join([
                f"- {r.get('client_email','?')} | {r.get('date','?')} | {r.get('subject','?')} | Acc: {r.get('account_no','Nil')}"
                for r in sample
            ])
            data_context = f"SYSTEM INJECTION: Searched '{intent.get('subject') or 'all'}'. Found exactly {count} matching records after filters.\nTop records:\n{sample_text}\n(If more exist, total is {count}. Use these exact details.)"
            session['dashboard_query'] = {"task_type": intent.get('task_type', 'standard'), "subject": intent.get('subject', ''), "date_filter": clean_date_filter(intent.get('date_filter', 'NONE'))}
            session.modified = True
        else:
            data_context = f"SYSTEM ERROR: IMAP failed. {results['error']}"
            
    elif intent.get("action") == "COMPARE":
        res1 = execute_imap_search(email_acc, password, 'standard', intent.get('subject_1', ''), 'NONE')
        res2 = execute_imap_search(email_acc, password, 'standard', intent.get('subject_2', ''), 'NONE')
        
        if res1['success'] and res2['success']:
            emails_1 = {r['client_email'].lower() for r in res1['records']}
            emails_2 = {r['client_email'].lower() for r in res2['records']}
            missing_emails = emails_1 - emails_2
            missing_list = list(missing_emails)
            
            data_context = f"SYSTEM INJECTION: Cross-reference complete. You found {len(emails_1)} emails in '{intent.get('subject_1')}' and {len(emails_2)} in '{intent.get('subject_2')}'. There are exactly {len(missing_emails)} emails in {intent.get('subject_1')} that are MISSING from {intent.get('subject_2')}. Here is the list to show the user: {', '.join(missing_list[:15])}"
        else:
            data_context = "SYSTEM ERROR: Could not fetch lists to compare."

    final_prompt = f"""
    You are an expert data agent. 
    INTERNAL DATA: {data_context}
    
    RULES:
    1. Act natural and human-like. NEVER announce your underlying AI model or provider.
    2. If you have data from SYSTEM INJECTION, use the exact emails, dates, and details provided. List them clearly when asked.
    3. If the user asks about a specific email, date range, or filter (e.g. starting with d/h/c), answer only from the injected records.
    4. If it was a single search, explicitly state the dataset is loaded onto their Dashboard.
    5. NEVER use placeholders or invent data.
    6. If INTERNAL DATA is empty, you have NOT searched the mailbox. Never state any email counts, totals or dates. If they ask about emails, ask for the subject and date range.
    7. Today is {today.strftime('%d %B %Y')}.
    
    Previous Chat:
    {chat_context}
    
    User: {user_message}
    """
    
    try:
        return query_llm(final_prompt)
    except Exception as e:
        return "I ran into a temporary issue reaching the analysis engine. Please try again in a moment."

# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    if 'username' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email_acc = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not username or not email_acc or not password:
            return render_template('register.html', error="All fields are required.")
        users = load_users()
        if username in users:
            return render_template('register.html', error="Username already taken.")
        # Verify IMAP credentials work
        try:
            with imap_connection(email_acc, password, select_inbox=False):
                pass
        except Exception:
            return render_template('register.html', error="Could not connect with that email/password.")
        users[username] = {"email": email_acc, "password": password}
        save_users(users)
        session['username'] = username
        session['email'] = email_acc
        session['password'] = password
        session['dashboard_query'] = None
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    users = load_users()
    user = users.get(username)
    if not user or user.get('password') != password:
        return render_template('login.html', error="Invalid username or password.")
    # Re-verify IMAP still works
    try:
        with imap_connection(user['email'], user['password'], select_inbox=False):
            pass
    except Exception:
        return render_template('login.html', error="Stored email credentials no longer work.")
    session['username'] = username
    session['email'] = user['email']
    session['password'] = user['password']
    session['dashboard_query'] = None
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session: return redirect(url_for('index'))
    
    total_inbox = 0
    match_count = 0
    records = []
    stat_label = "System Ready"
    search_term = ""
    task_type = "standard"
    ai_message = "Awaiting manual query or AI chat instructions."
    target_subject = ""
    date_filter = "NONE"
    action = "search"

    active_query = session.get('dashboard_query')
    
    if request.method == 'POST':
        search_term = request.form.get('search_term', '')
        action = request.form.get('action', 'search')
        try:
            prompt = f"Extract task_type (standard or client_tally), subject, and date_filter for this request as JSON only: '{search_term}'"
            parsed = parse_llm_json(query_llm(prompt))
            task_type = parsed.get("task_type", "standard")
            target_subject = parsed.get("subject", search_term)
            date_filter = parsed.get("date_filter", "NONE")
        except Exception:
            # Robust fallback: treat the raw term as the subject so results still appear
            task_type = "standard"
            target_subject = search_term.strip()
            date_filter = "NONE"
            ai_message = "Using direct subject match (analysis engine temporarily limited)."
            
    elif active_query:
        task_type = active_query['task_type']
        target_subject = active_query['subject']
        date_filter = active_query['date_filter']
        search_term = f"AI Synced: {target_subject}"
        ai_message = "Dashboard synchronized with your recent AI Chat inquiry."
        action = "search"
    else:
        try:
            with imap_connection(session['email'], session['password']) as mail:
                _, all_msgs = mail.search(None, 'ALL')
                total_inbox = len(all_msgs[0].split()) if all_msgs[0] else 0
        except: pass
        return render_template('dashboard.html', total_inbox=total_inbox, match_count=0, records=[], search_term="", task_type="standard", stat_label="Awaiting Request", ai_message="System ready.")

    results = execute_imap_search(session['email'], session['password'], task_type, target_subject, date_filter)
    
    if results['success']:
        total_inbox = results['total_inbox']
        match_count = results['match_count']
        records = results['records']
        stat_label = "Unique Clients" if task_type == "client_tally" else f"Matches: {target_subject}"
    else:
        ai_message = f"Search issue: {results.get('error', 'Unknown')}"

    if request.method == 'POST' and action == "export":
        output = io.StringIO()
        writer = csv.writer(output)
        if task_type == 'client_tally':
            writer.writerow(['No', 'Unique Client Email', 'Status'])
            for idx, r in enumerate(records, start=1):
                writer.writerow([idx, r['client_email'], 'Verified Client'])
        else:
            writer.writerow(['No', 'Client Email', 'Account Number', 'Subject', 'Date'])
            for idx, r in enumerate(records, start=1):
                writer.writerow([idx, r['client_email'], r['account_no'], r['subject'], r['date']])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-disposition": "attachment; filename=Analysis_AI_Report.csv"})

    return render_template('dashboard.html', total_inbox=total_inbox, match_count=match_count, records=records, search_term=search_term, task_type=task_type, stat_label=stat_label, ai_message=ai_message)

@app.route('/chat', methods=['GET'])
def chat_page():
    if 'username' not in session: return redirect(url_for('index'))
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'username' not in session: return jsonify({"error": "Unauthorized"}), 401
    user_message = request.json.get('message', '')
    if not user_message: return jsonify({"reply": "I didn't catch that."})
    
    username = session['username']
    history = load_memory(username)
    
    reply = process_agentic_chat(user_message, history, session['email'], session['password'])
    
    history.append({"role": "User", "text": user_message})
    history.append({"role": "Analysis AI", "text": reply})
    save_memory(username, history[-20:])
    
    return jsonify({"reply": reply})

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    if 'username' not in session: return jsonify([])
    return jsonify(load_memory(session['username']))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
