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
    page_title="IndoGuide",
    page_icon="🇮🇩",
    layout="centered"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = logger.create_session(st.session_state.session_id)
    st.session_state.llm_client = LLMClient()
    st.session_state.system_prompt = SYSTEM_PROMPT


def restart_conversation():
    """Start a new conversation session"""
    # Save current session
    logger.save_session(st.session_state.session_log)
    
    # Create new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = logger.create_session(st.session_state.session_id)
    st.session_state.llm_client.reset_conversation()


# Title and session info

# Session info and restart button in columns
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <style>
        .info-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .info-button-wrapper {
            position: relative;
            display: inline-block;
        }
        .info-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background-color: #f0f2f6;
            color: #31333F;
            font-size: 14px;
            font-weight: bold;
            cursor: help;
            border: 1px solid #d0d0d0;
            transition: all 0.2s;
        }
        .info-button:hover {
            background-color: #e0e2e6;
            border-color: #a0a0a0;
        }
        .tooltip-text {
            visibility: hidden;
            width: 400px;
            background-color: #262730;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 12px 16px;
            position: absolute;
            z-index: 1000;
            top: 125%;
            left: 50%;
            margin-left: -200px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.5;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .tooltip-text::after {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent #262730 transparent;
        }
        .info-button-wrapper:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        </style>
        <div class="info-container">
            <h1 style="margin: 0;">🇮🇩 IndoGuide</h1>
            <div class="info-button-wrapper">
                <span class="info-button">i</span>
                <span class="tooltip-text">IndoGuide is your smart travel companion designed to make exploring Indonesia effortless. Ask away information on must-see destinations, visas, transportation, safety, and local etiquettes, so you can travel with confidence. Whether you're planning your itinerary or navigating on the go, IndoGuide helps you experience Indonesia like a pro!</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
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

# Display chat messages from LLM client
for message in st.session_state.llm_client.get_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Get user timestamp
    user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display user message
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
        
        # Show thinking indicator while waiting for response
        with st.spinner("Thinking..."):
            # Stream the response (auto_add_messages=True by default)
            for chunk in st.session_state.llm_client.chat_stream(
                user_message=prompt,
                system_prompt=st.session_state.system_prompt
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Get bot timestamp (when streaming finished)
    bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log bot turn
    logger.add_turn(
        st.session_state.session_log,
        role="bot",
        text=full_response,
        timestamp=bot_timestamp
    )
    
    # Auto-save session after each turn
    logger.save_session(st.session_state.session_log)
