import streamlit as st
import uuid
from datetime import datetime
from llm_client import LLMClient
from logger import DialogueLogger
from config import SYSTEM_PROMPT


# Initialize logger
logger = DialogueLogger()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = logger.create_session(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.llm_client = LLMClient()
    st.session_state.system_prompt = SYSTEM_PROMPT


def restart_conversation():
    """Start a new conversation session"""
    # Save current session
    logger.save_session(st.session_state.session_log)
    
    # Create new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = logger.create_session(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.llm_client.reset_conversation()


# Title and session info

# Session info and restart button in columns
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 AI Chatbot")
    st.caption(f"**Session ID:** `{st.session_state.session_id}`")
with col2:
    # Add custom CSS to vertically align to bottom and right-align the button
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-of-type(2) {
            display: flex;
            align-items: flex-end;
            justify-content: flex-end;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Restart", use_container_width=True):
        restart_conversation()
        st.rerun()

st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Get user timestamp
    user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Log user turn
    logger.add_turn(
        st.session_state.session_log,
        role="user",
        text=prompt,
        timestamp=user_timestamp
    )
    
    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in st.session_state.llm_client.chat_stream(
            user_message=prompt,
            system_prompt=st.session_state.system_prompt
        ):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Get bot timestamp (when streaming finished)
    bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Log bot turn
    logger.add_turn(
        st.session_state.session_log,
        role="bot",
        text=full_response,
        timestamp=bot_timestamp
    )
    
    # Auto-save session after each turn
    logger.save_session(st.session_state.session_log)
