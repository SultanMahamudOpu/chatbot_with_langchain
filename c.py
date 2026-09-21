import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from dotenv import load_dotenv

# =========================================================
# ENVIRONMENT
# =========================================================
load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Infera AI | AI Assistant",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS (UI ENHANCEMENTS)
# =========================================================
def apply_custom_css():
    st.markdown("""
        <style>
        /* Gradient text for the main title */
        .gradient-text {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 0rem;
        }
        /* Subtitle styling */
        .sub-title {
            text-align: center;
            color: #888;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        /* Button hover effects */
        .stButton>button {
            border-radius: 8px;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #00C9FF;
        }
        /* Welcome cards */
        .welcome-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            height: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# =========================================================
# SESSION STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_style" not in st.session_state:
    st.session_state.response_style = "Balanced"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

# =========================================================
# SYSTEM PROMPT
# =========================================================
def build_system_prompt(style):
    style_instruction = {
        "Concise": "Keep responses concise, direct, and focused. Avoid unnecessary explanations unless they are important.",
        "Balanced": "Provide clear, useful, and well-structured responses with a balanced level of detail.",
        "Detailed": "Provide thorough explanations with useful context, examples, reasoning, and important edge cases when appropriate."
    }

    return f"""
    You are Infera AI, an intelligent, reliable, and context-aware AI assistant.
    Your goal is to understand the user's actual intent and provide the most useful answer possible.
    
    RESPONSE STYLE:
    {style_instruction[style]}
    
    GENERAL BEHAVIOR:
    - If the user writes in Bangla, respond in Bangla naturally unless English is requested.
    - Use Markdown for readability (headings, bullets, bold text, code blocks).
    - Never fabricate facts or pretend to run code if you haven't.
    """

# =========================================================
# MODEL INITIALIZATION
# =========================================================
@st.cache_resource
def get_model(temperature):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
    )

model = get_model(st.session_state.temperature)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ **Settings**")
    st.caption("Customize your Infera AI experience")
    st.divider()

    st.markdown("**🎭 Response Style**")
    response_style = st.selectbox(
        "Response Style",
        options=["Balanced", "Concise", "Detailed"],
        index=["Balanced", "Concise", "Detailed"].index(st.session_state.response_style),
        label_visibility="collapsed",
    )
    st.session_state.response_style = response_style

    st.markdown("**🧠 Creativity Level**")
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1,
        help="Lower values produce more factual responses. Higher values make it more creative.",
        label_visibility="collapsed"
    )
    
    if temperature != st.session_state.temperature:
        st.session_state.temperature = temperature
        model = get_model(temperature)

    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Powered by LangChain & Groq ⚡")

# =========================================================
# HEADER & EMPTY STATE (WELCOME SCREEN)
# =========================================================
if not st.session_state.chat_history:
    st.markdown("<h1 class='gradient-text'>Infera AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your intelligent, reasoning-powered conversation partner.</p>", unsafe_allow_html=True)
    
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='welcome-card'>📝<br><b>Summarize</b><br><span style='font-size:0.8rem;color:gray;'>Make complex topics simple</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='welcome-card'>💻<br><b>Code & Debug</b><br><span style='font-size:0.8rem;color:gray;'>Write C, C++, Python and more</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='welcome-card'>🧠<br><b>Brainstorm</b><br><span style='font-size:0.8rem;color:gray;'>Generate creative ideas</span></div>", unsafe_allow_html=True)
    
    st.write("")
    st.divider()
else:
    # Minimal header when chatting
    st.markdown("### 🌌 **Infera AI**")
    st.divider()

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================
# Custom Avatars
USER_AVATAR = "🧑‍🚀"
BOT_AVATAR = "✨"

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(message.content)

# =========================================================
# CHAT INPUT & PROCESSING
# =========================================================
user_input = st.chat_input("Message Infera AI...")

if user_input:
    if user_input.strip().lower() == "exit":
        st.stop()

    system_message = SystemMessage(content=build_system_prompt(st.session_state.response_style))
    human_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(human_message)

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            messages = [system_message, *st.session_state.chat_history]
            with st.spinner("Thinking..."):
                response = st.write_stream(model.stream(messages))
            
            st.session_state.chat_history.append(AIMessage(content=str(response)))
        
        except Exception as e:
            st.error("Something went wrong while generating the response.")
            st.caption(f"Error: {str(e)}")
            if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
                st.session_state.chat_history.pop()

# =========================================================
# ACTION BUTTONS (REGENERATE & EXPORT)
# =========================================================
if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], AIMessage):
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Regenerate Response", use_container_width=True):
            st.session_state.chat_history.pop() # Remove previous AI response
            system_message = SystemMessage(content=build_system_prompt(st.session_state.response_style))
            messages = [system_message, *st.session_state.chat_history]
            
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                try:
                    with st.spinner("Thinking..."):
                        response = st.write_stream(model.stream(messages))
                    st.session_state.chat_history.append(AIMessage(content=str(response)))
                    st.rerun()
                except Exception as e:
                    st.error("Failed to regenerate the response.")
                    st.caption(f"Error: {str(e)}")

    with col2:
        conversation_text = ""
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                conversation_text += f"USER:\n{message.content}\n\n"
            elif isinstance(message, AIMessage):
                conversation_text += f"Infera AI:\n{message.content}\n\n"

        st.download_button(
            label="📥 Export Chat",
            data=conversation_text,
            file_name="infera_ai_history.txt",
            mime="text/plain",
            use_container_width=True,
        )