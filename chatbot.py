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
    page_title="NovaChat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


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
        "Concise": """
Keep responses concise, direct, and focused.
Avoid unnecessary explanations unless they are important.
""",

        "Balanced": """
Provide clear, useful, and well-structured responses with
a balanced level of detail.
""",

        "Detailed": """
Provide thorough explanations with useful context, examples,
reasoning, and important edge cases when appropriate.
"""
    }

    return f"""
You are NovaChat, an intelligent, reliable, and context-aware
AI assistant.

Your goal is to understand the user's actual intent and provide
the most useful answer possible.

GENERAL BEHAVIOR:
- Understand the user's intent before responding.
- Use conversation history to maintain context.
- Never ask for information that the user has already provided.
- Adapt your explanation to the user's knowledge level.
- Be natural, professional, friendly, and direct.
- Match the user's language naturally.
- If the user writes in Bangla, respond in Bangla unless English
  is clearly more appropriate.
- Use Markdown when it improves readability.
- Use headings, bullets, numbered lists, tables, and code blocks
  when they make the answer clearer.

ACCURACY AND HONESTY:
- Never fabricate facts, statistics, citations, research findings,
  code behavior, results, or sources.
- Never claim that you browsed the web, executed code, accessed
  a file, or performed an action unless you actually did so.
- Clearly communicate uncertainty when necessary.
- Correct important mistakes politely.
- Distinguish facts, assumptions, examples, and opinions.

PROGRAMMING:
- Identify the likely root cause before proposing a solution.
- Provide clean and practical code when requested.
- Preserve the user's framework and approach whenever possible.
- Explain important code changes clearly.
- Consider common errors and edge cases.
- Never claim that code was tested unless it was actually tested.

RESEARCH AND ACADEMIC WORK:
- Use precise academic terminology.
- Never invent references, datasets, results, methodologies,
  or statistical findings.
- Distinguish established facts from assumptions.
- Maintain scientific accuracy and reproducibility.
- Mention limitations when they are relevant.

PROBLEM SOLVING:
- Break complex problems into logical steps.
- Focus on actionable solutions.
- When multiple approaches are possible, explain the key
  differences without unnecessary complexity.

RESPONSE STYLE:
{style_instruction[style]}

FINAL QUALITY CHECK:
Before responding, silently verify that the answer is:
1. Accurate
2. Relevant
3. Clear
4. Useful
5. Appropriately concise

Priority:
Accuracy > Relevance > Clarity > Usefulness > Brevity
"""


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def get_model(temperature):

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
    )


model = get_model(
    st.session_state.temperature
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.divider()

    # -----------------------------------------------------
    # Response Style
    # -----------------------------------------------------

    st.subheader("Response Style")

    response_style = st.selectbox(
        "Response Style",
        options=[
            "Balanced",
            "Concise",
            "Detailed",
        ],
        index=[
            "Balanced",
            "Concise",
            "Detailed",
        ].index(
            st.session_state.response_style
        ),
        label_visibility="collapsed",
    )

    st.session_state.response_style = response_style

    # -----------------------------------------------------
    # Creativity
    # -----------------------------------------------------

    st.subheader("Creativity")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help=(
            "Lower values produce more consistent responses. "
            "Higher values produce more creative responses."
        ),
    )

    if temperature != st.session_state.temperature:

        st.session_state.temperature = temperature

        model = get_model(
            temperature
        )

    # -----------------------------------------------------
    # Clear Conversation
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "### ✦ AI • Reasoning • Conversation"
)

st.title("🤖 NovaChat")

st.divider()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.markdown(
                message.content
            )

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.markdown(
                message.content
            )


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Message NovaChat..."
)


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Exit Command
    # -----------------------------------------------------

    if user_input.strip().lower() == "exit":

        st.stop()

    # -----------------------------------------------------
    # System Message
    # -----------------------------------------------------

    system_message = SystemMessage(
        content=build_system_prompt(
            st.session_state.response_style
        )
    )

    # -----------------------------------------------------
    # User Message
    # -----------------------------------------------------

    human_message = HumanMessage(
        content=user_input
    )

    st.session_state.chat_history.append(
        human_message
    )

    # -----------------------------------------------------
    # Display User Message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    # -----------------------------------------------------
    # Generate AI Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            messages = [
                system_message,
                *st.session_state.chat_history,
            ]

            response = st.write_stream(
                model.stream(messages)
            )

            # Save AI response
            st.session_state.chat_history.append(
                AIMessage(
                    content=str(response)
                )
            )

        except Exception as e:

            st.error(
                "Something went wrong while generating the response."
            )

            st.caption(
                f"Error: {str(e)}"
            )

            # Remove failed user message
            if (
                st.session_state.chat_history
                and isinstance(
                    st.session_state.chat_history[-1],
                    HumanMessage,
                )
            ):

                st.session_state.chat_history.pop()


# =========================================================
# ACTION BUTTONS
# =========================================================

if (
    st.session_state.chat_history
    and isinstance(
        st.session_state.chat_history[-1],
        AIMessage,
    )
):

    st.divider()

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Regenerate Response
    # -----------------------------------------------------

    with col1:

        if st.button(
            "🔄 Regenerate",
            use_container_width=True,
        ):

            # Remove previous AI response
            st.session_state.chat_history.pop()

            system_message = SystemMessage(
                content=build_system_prompt(
                    st.session_state.response_style
                )
            )

            messages = [
                system_message,
                *st.session_state.chat_history,
            ]

            with st.chat_message("assistant"):

                try:

                    response = st.write_stream(
                        model.stream(messages)
                    )

                    st.session_state.chat_history.append(
                        AIMessage(
                            content=str(response)
                        )
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Failed to regenerate the response."
                    )

                    st.caption(
                        f"Error: {str(e)}"
                    )

    # -----------------------------------------------------
    # Export Chat
    # -----------------------------------------------------

    with col2:

        conversation_text = ""

        for message in st.session_state.chat_history:

            if isinstance(message, HumanMessage):

                conversation_text += (
                    f"USER:\n"
                    f"{message.content}\n\n"
                )

            elif isinstance(message, AIMessage):

                conversation_text += (
                    f"NOVACHAT:\n"
                    f"{message.content}\n\n"
                )

        st.download_button(
            label="📥 Export Chat",
            data=conversation_text,
            file_name="novachat_conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )