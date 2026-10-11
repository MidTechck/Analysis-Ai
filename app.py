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

load_dotenv()

app = Flask(__name__)
app.secret_key = "sovetlink_admin_secure_key_2026"

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
                preferred = ["models/gemini-1.5-flash", "models/gemini-1.5-flash-8b", "models/gemini-1.5-pro", "models/gemini-2.0-flash"]
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
    return json.loads(cleaned)

# --- STARTUP ---
print("\n" + "="*50)
print("SYSTEM BOOT DIAGNOSTICS & MULTI-PROVIDER LLM:")
discover_providers()
print("="*50 + "\n")

# --- PERMANENT MEMORY ENGINE ---
def get_memory_file(email_acc):
    safe_email = email_acc.replace('@', '_').replace('.', '_')
    return f"memory_{safe_email}.json"

def load_memory(email_acc):
    file = get_memory_file(email_acc)
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(email_acc, history):
    file = get_memory_file(email_acc)
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
    if task_type == "client_tally":
        return f'(ALL {date_filter})' if date_filter != "NONE" else 'ALL'
    else:
        if date_filter != "NONE":
            return f'(SUBJECT "{target_subject}" {date_filter})'
        return f'SUBJECT "{target_subject}"'

def execute_imap_search(email_acc, password, task_type, target_subject, date_filter):
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
                            parsed_date = email.utils.parsedate_to_datetime(raw_date)
                            clean_date = parsed_date.strftime("%d %b %Y, %H:%M") if parsed_date else raw_date
                            
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

def process_agentic_chat(user_message, history, email_acc, password):
    chat_context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history[-8:]])
    
    router_prompt = f"""
    Analyze the user's latest message based on chat history.
    
    If the user wants to pull a SINGLE dataset, output ONLY:
    {{"action": "SEARCH", "task_type": "standard", "subject": "Exact Subject (e.g. Order confirmed)", "date_filter": "NONE"}}
    
    If the user wants to CROSS-REFERENCE two datasets (e.g., check who paid but didn't order), output ONLY:
    {{"action": "COMPARE", "subject_1": "Primary subject (e.g. payment processed)", "subject_2": "Missing subject (e.g. order confirmed)"}}
    
    If general chat, output ONLY:
    {{"action": "CHAT"}}
    
    History:
    {chat_context}
    User latest: {user_message}
    """
    
    try:
        raw_intent = query_llm(router_prompt)
        intent = parse_llm_json(raw_intent)
    except Exception:
        intent = {"action": "CHAT"}

    data_context = ""
    if intent.get("action") == "SEARCH":
        results = execute_imap_search(email_acc, password, intent.get('task_type', 'standard'), intent.get('subject', ''), intent.get('date_filter', 'NONE'))
        if results['success']:
            count = results['match_count']
            data_context = f"SYSTEM INJECTION: You just searched for '{intent.get('subject')}'. The live server confirmed exactly {count} matches."
            session['dashboard_query'] = {"task_type": intent.get('task_type', 'standard'), "subject": intent.get('subject', ''), "date_filter": intent.get('date_filter', 'NONE')}
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
    2. If you have data from SYSTEM INJECTION, give the user the precise insights requested. 
    3. If resolving a CROSS-REFERENCE (COMPARE), explicitly list the specific emails you found so the user can take action.
    4. If it was a single search, explicitly state the dataset is loaded onto their Dashboard.
    5. NEVER use placeholders.
    
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
    if 'email' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email_acc = request.form['email']
    password = request.form['password']
    try:
        with imap_connection(email_acc, password, select_inbox=False):
            pass
        session['email'] = email_acc
        session['password'] = password
        session['dashboard_query'] = None
        return redirect(url_for('dashboard'))
    except Exception:
        return render_template('login.html', error="Invalid credentials.")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'email' not in session: return redirect(url_for('index'))
    
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
    if 'email' not in session: return redirect(url_for('index'))
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'email' not in session: return jsonify({"error": "Unauthorized"}), 401
    user_message = request.json.get('message', '')
    if not user_message: return jsonify({"reply": "I didn't catch that."})
    
    email_acc = session['email']
    history = load_memory(email_acc)
    
    reply = process_agentic_chat(user_message, history, email_acc, session['password'])
    
    history.append({"role": "User", "text": user_message})
    history.append({"role": "Analysis AI", "text": reply})
    save_memory(email_acc, history[-20:])
    
    return jsonify({"reply": reply})

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    if 'email' not in session: return jsonify([])
    return jsonify(load_memory(session['email']))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
