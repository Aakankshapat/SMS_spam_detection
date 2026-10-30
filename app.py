
import os
import sqlite3
import pickle
from datetime import datetime

import streamlit as st

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detection Simulator",
    page_icon="📱",
    layout="wide"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(APP_DIR, "vectorizer.pkl")
DB_PATH = os.path.join(APP_DIR, "sms_simulator.db")


# --------------------------------------------------
# 2. IMPROVED UI TEXT SIZE AND STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 1.9rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        font-size: 1.08rem;
        line-height: 1.65;
    }

    .sms-body {
        font-size: 1.25rem;
        line-height: 1.8;
        font-weight: 500;
        overflow-wrap: anywhere;
        padding: 10px 0;
    }

    .sender-name {
        font-size: 1.3rem;
        font-weight: 750;
    }

    .ai-label {
        font-size: 1.1rem;
        font-weight: 800;
        padding: 9px 12px;
        border-radius: 8px;
        margin: 10px 0;
    }

    .spam-label {
        background-color: #ffebeb;
        color: #a00000;
        border: 1px solid #e6a0a0;
    }

    .ham-label {
        background-color: #e6f7ec;
        color: #126b35;
        border: 1px solid #91d4a6;
    }

    .user-label {
        background-color: #e8f1ff;
        color: #174ea6;
        border: 1px solid #9bbcf5;
    }

    .stButton > button {
        min-height: 48px;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
    }

    @media (max-width: 640px) {
        h1 {
            font-size: 2rem !important;
        }

        .sms-body {
            font-size: 1.12rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    model, vectorizer = load_model()
    model_loaded = True
    model_error = ""
except Exception as e:
    model_loaded = False
    model_error = str(e)


# --------------------------------------------------
# 4. DATABASE AND MIGRATION
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                body TEXT NOT NULL,
                prediction TEXT NOT NULL,
                spam_probability REAL,
                folder TEXT NOT NULL DEFAULT 'Inbox',
                received_at TEXT NOT NULL,
                user_decision TEXT NOT NULL DEFAULT 'Unreviewed'
            )
        """)

        # Add the new column to an existing database if necessary.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }

        if "user_decision" not in columns:
            conn.execute("""
                ALTER TABLE messages
                ADD COLUMN user_decision TEXT
                NOT NULL DEFAULT 'Unreviewed'
            """)


def add_message(sender, body, prediction, probability):
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # AI predictions never automatically move a message to Spam.
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO messages (
                sender, body, prediction, spam_probability,
                folder, received_at, user_decision
            )
            VALUES (?, ?, ?, ?, 'Inbox', ?, 'Unreviewed')
        """, (sender, body, prediction, probability, timestamp))


def get_messages(folder):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("""
            SELECT * FROM messages
            WHERE folder = ?
            ORDER BY id DESC
        """, (folder,)).fetchall()


def update_message(message_id, folder, decision):
    with get_connection() as conn:
        conn.execute("""
            UPDATE messages
            SET folder = ?, user_decision = ?
            WHERE id = ?
        """, (folder, decision, message_id))


def delete_message(message_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM messages WHERE id = ?",
            (message_id,)
        )


initialize_database()


# --------------------------------------------------
# 5. CLASSIFICATION
# --------------------------------------------------

def classify_sms(message):
    features = vectorizer.transform([message])
    prediction = str(model.predict(features)[0]).strip().lower()

    if prediction not in ("spam", "ham"):
        raise ValueError(
            f"Unexpected model label: {prediction}"
        )

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [
            str(label).strip().lower()
            for label in model.classes_
        ]

        if "spam" in classes:
            probability = round(
                float(probabilities[classes.index("spam")]) * 100,
                2
            )

    return prediction, probability


# --------------------------------------------------
# 6. APP HEADER
# --------------------------------------------------

st.title("📱 SMS Spam Detection Simulator")

st.markdown(
    "Your AI model checks incoming messages. "
    "**You decide whether a message is spam or legitimate.**"
)

if not model_loaded:
    st.error(f"Could not load model files: {model_error}")
    st.info(
        "Keep app.py, model.pkl, and vectorizer.pkl "
        "in the same project folder."
    )
    st.stop()

st.success("✅ Trained model and vectorizer loaded successfully.")


# --------------------------------------------------
# 7. SIDEBAR FOLDERS
# --------------------------------------------------

inbox_count = len(get_messages("Inbox"))
spam_count = len(get_messages("Spam"))

st.sidebar.title("📂 Message Folders")

folder = st.sidebar.radio(
    "Select folder",
    ["Inbox", "Spam"],
    format_func=lambda item: (
        f"📥 Inbox ({inbox_count})"
        if item == "Inbox"
        else f"🚨 Spam ({spam_count})"
    )
)

st.sidebar.divider()
st.sidebar.markdown("**How it works**")
st.sidebar.caption(
    "1. AI predicts the message category.\n\n"
    "2. New messages stay in Inbox.\n\n"
    "3. You choose the correct folder."
)


# --------------------------------------------------
# 8. RECEIVE SMS
# --------------------------------------------------

st.header("📩 Simulate  SMS")

with st.form("incoming_sms_form", clear_on_submit=True):
    sender = st.text_input(
        "Sender name or number",
        placeholder="e.g. Courier, Friend, +91..."
    )

    body = st.text_area(
        "SMS message",
        placeholder="Type or paste your SMS here...",
        height=130
    )

    submitted = st.form_submit_button(
        "📨 Send SMS",
        use_container_width=True
    )

    if submitted:
        sender = sender.strip() or "Unknown Sender"
        body = body.strip()

        if not body:
            st.warning("Please enter a message.")
        else:
            try:
                prediction, probability = classify_sms(body)
                add_message(sender, body, prediction, probability)

                st.session_state["last_received"] = {
                    "prediction": prediction,
                    "probability": probability
                }

                st.rerun()

            except Exception as e:
                st.error(f"Could not classify the SMS: {e}")


# --------------------------------------------------
# 9. LATEST AI RESULT
# --------------------------------------------------

last = st.session_state.get("last_received")

if last:
    if last["prediction"] == "spam":
        st.warning(
            "🚨 AI DETECTION: POSSIBLE SPAM\n\n"
            "This is a model prediction, not a final decision. "
            "The message remains in Inbox until you move it."
        )
    else:
        st.success(
            "✅ AI DETECTION: LIKELY LEGITIMATE\n\n"
            "The message remains in Inbox. You can still move it "
            "to Spam if you disagree with the prediction."
        )

    if last["probability"] is not None:
        st.metric(
            "Estimated spam probability",
            f"{last['probability']:.2f}%"
        )


# --------------------------------------------------
# 10. DISPLAY MESSAGES
# --------------------------------------------------

st.divider()

if folder == "Inbox":
    st.header("📥 Inbox")
    st.caption(
        "All new messages appear here first, including messages "
        "the model has flagged as possible spam."
    )
else:
    st.header("🚨 Spam Folder")
    st.caption(
        "Messages you have chosen to move to Spam appear here."
    )

messages = get_messages(folder)

if not messages:
    st.info(
        "No messages in this folder yet. "
        "Receive a new SMS above to get started."
    )


# --------------------------------------------------
# 11. MESSAGE CARDS AND USER CONTROLS
# --------------------------------------------------

for msg in messages:
    message_id = msg["id"]
    prediction = msg["prediction"].lower()
    decision = msg["user_decision"] or "Unreviewed"

    with st.container(border=True):
        st.markdown(
            f'<div class="sender-name">👤 '
            f'{msg["sender"]}</div>',
            unsafe_allow_html=True
        )

        st.caption(f'🕒 {msg["received_at"]}')

        st.markdown(
            f'<div class="sms-body">{msg["body"]}</div>',
            unsafe_allow_html=True
        )

        # Show original AI prediction separately from user decision.
        if prediction == "spam":
            st.markdown(
                '<div class="ai-label spam-label">'
                '🚨 AI DETECTION: SPAM</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ai-label ham-label">'
                '✅ AI DETECTION: LIKELY LEGITIMATE</div>',
                unsafe_allow_html=True
            )

        if msg["spam_probability"] is not None:
            st.caption(
                f'AI estimated spam probability: '
                f'{msg["spam_probability"]:.2f}%'
            )

        if decision == "Not Spam":
            st.markdown(
                '<div class="ai-label user-label">'
                '✅ USER DECISION: MARKED AS NOT SPAM</div>',
                unsafe_allow_html=True
            )
        elif decision == "Spam":
            st.markdown(
                '<div class="ai-label user-label">'
                '🚨 USER DECISION: CONFIRMED AS SPAM</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Choose your action:**")

        if folder == "Inbox":
            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "✅ Mark as Not Spam",
                    key=f"not_spam_{message_id}",
                    use_container_width=True
                ):
                    update_message(
                        message_id, "Inbox", "Not Spam"
                    )
                    st.toast("Message marked as Not Spam.")
                    st.rerun()

            with col2:
                if st.button(
                    "🚨 Move to Spam",
                    key=f"move_spam_{message_id}",
                    use_container_width=True
                ):
                    update_message(
                        message_id, "Spam", "Spam"
                    )
                    st.toast("Message moved to Spam.")
                    st.rerun()

        else:
            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "✅ Mark as Not Spam",
                    key=f"restore_{message_id}",
                    use_container_width=True
                ):
                    update_message(
                        message_id, "Inbox", "Not Spam"
                    )
                    st.toast("Message restored to Inbox.")
                    st.rerun()

            with col2:
                if st.button(
                    "🗑️ Delete Message",
                    key=f"delete_{message_id}",
                    use_container_width=True
                ):
                    delete_message(message_id)
                    st.toast("Message deleted.")
                    st.rerun()
