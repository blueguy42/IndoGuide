import streamlit as st
import uuid
from datetime import datetime
from llm_client import LLMClient
from logger import DialogueLogger
from rag_system import RAGSystem
from config import SYSTEM_PROMPT, MODEL_NAME, RAG_NAME_TO_ID, STARTER_MESSAGE


# Initialize logger
logger = DialogueLogger()

# Page configuration
st.set_page_config(
    page_title="IndoGuide",
    page_icon="🇮🇩",
    layout="centered"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.rag_config = 1  # Default to baseline
    st.session_state.session_log = None  # Will be created on first user input
    st.session_state.session_started = False  # Track if user has sent first message
    st.session_state.llm_client = LLMClient(model=MODEL_NAME)
    st.session_state.llm_client.add_assistant_message(STARTER_MESSAGE)
    st.session_state.rag_system = None  # Will be initialized based on config


def restart_conversation():
    """Start a new conversation session"""
    # Save current session only if it was started
    if st.session_state.session_started:
        logger.save_session(st.session_state.session_log)
    
    # Create new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = None  # Will be created on first user input
    st.session_state.session_started = False
    st.session_state.llm_client.reset_conversation()
    st.session_state.llm_client.add_assistant_message(STARTER_MESSAGE)


# Title and session info

# Session info and restart button in columns
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
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
    # Custom CSS loaded globally handles alignment

    if st.button("Restart", use_container_width=True):
        restart_conversation()
        st.rerun()

st.divider()

# Sidebar for RAG configuration
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    
    selected_config = st.radio(
        "Select RAG Configuration:",
        options=list(RAG_NAME_TO_ID.keys()),
        index=st.session_state.rag_config - 1,
        help="Choose how retrieved context is ranked before being used."
    )
    
    new_config = RAG_NAME_TO_ID[selected_config]
    
    # Reinitialize RAG system if configuration changed
    if new_config != st.session_state.rag_config or st.session_state.rag_system is None:
        # If config changed (not initial load), restart the session
        if st.session_state.rag_system is not None:
            st.warning("RAG configuration changed. Restarting session...")
            # Update config BEFORE restarting to prevent radio button from reverting
            st.session_state.rag_config = new_config
            restart_conversation()
            with st.spinner(f"Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(config=new_config)
            st.rerun()  # Refresh to clear messages and show clean interface
        else:
            # Initial load - just set the config
            st.session_state.rag_config = new_config
            with st.spinner(f"Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(config=new_config)

# Display chat messages from LLM client
for message in st.session_state.llm_client.get_messages():
    avatar = "icon_assistant.png" if message["role"] == "assistant" else "icon_user.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Create session log on first user input
    if not st.session_state.session_started:
        st.session_state.session_log = logger.create_session(st.session_state.session_id, st.session_state.rag_config)
        st.session_state.session_started = True
    
    # Get user timestamp
    user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display user message
    with st.chat_message("user", avatar="icon_user.png"):
        st.markdown(prompt)
    
    # Retrieve relevant context using RAG
    with st.spinner("Retrieving relevant information..."):
        retrieved_snippets = st.session_state.rag_system.retrieve(prompt)
        context = st.session_state.rag_system.format_context(retrieved_snippets)
    
    # Inject context into system prompt
    augmented_prompt = context + "\n" + SYSTEM_PROMPT
    
    # If this is the first real turn (only starter message in history), inform the model
    if len(st.session_state.llm_client.messages) == 1:
        print("Informing model about starter message...")
        augmented_prompt += f"\n\n[Context: You have just started the conversation with this greeting, so do not introduce yourself again: '{STARTER_MESSAGE}']"
    
    # Log user turn with retrieved snippets
    logger.add_turn(
        st.session_state.session_log,
        speaker="user",
        utterance=prompt,
        timestamp=user_timestamp,
        retrieved_snippets=retrieved_snippets
    )
    
    # Get bot response
    with st.chat_message("assistant", avatar="icon_assistant.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show thinking indicator while waiting for response
        with st.spinner("Thinking..."):
            # Stream the response with augmented prompt
            for chunk in st.session_state.llm_client.chat_stream(
                user_message=prompt,
                system_prompt=augmented_prompt
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Get bot timestamp (when streaming finished)
    bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log bot turn
    logger.add_turn(
        st.session_state.session_log,
        speaker="assistant",
        utterance=full_response,
        timestamp=bot_timestamp
    )
    
    # Auto-save session after each turn
    logger.save_session(st.session_state.session_log)
