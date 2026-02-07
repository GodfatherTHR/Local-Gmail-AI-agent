import os
import json
import base64
from typing import TypedDict, List
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# Update scopes to include sending permissions
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]
CHAT_MODEL = 'qwen2.5:1.5b'

# --- Streamlit Setup ---
st.set_page_config(page_title="AI Agent with Gmail", page_icon="🤖")
st.title("🤖 Gmail AI Agent")

# --- Google Auth & Service ---
def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None
    
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # Check if current creds have all required scopes
            if not set(SCOPES).issubset(set(creds.scopes)):
                st.warning("Updating permissions... Re-authentication required.")
                creds = None
        except Exception:
            creds = None
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None # Force re-login if refresh fails
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

# --- Tools ---

@tool
def list_unread_emails():
    """Return a bullet list of every UNREAD message's ID, subject, date and sender"""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me', q='is:unread', maxResults=10
        ).execute()
        messages = results.get('messages', [])
        
        if not messages:
            return 'You have no unread messages.'
        
        unread_emails = []
        for msg in messages:
            message = service.users().messages().get(
                userId='me', id=msg['id'], format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
            
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            unread_emails.append({
                'id': msg['id'],
                'subject': subject,
                'sender': sender,
                'date': date
            })
        return json.dumps(unread_emails, indent=2)
    except Exception as e:
        return f'Error fetching emails: {str(e)}'

@tool
def summarize_email(message_id: str):
    """Summarize a single email given its message ID."""
    try:
        service = get_gmail_service()
        message = service.users().messages().get(
            userId='me', id=message_id, format='full'
        ).execute()
        
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        
        body = ''
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        elif 'body' in message['payload'] and 'data' in message['payload']['body']:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
            
        if not body:
            return f"Could not extract body for email ID {message_id}"
            
        return f"Subject: {subject}\nSender: {sender}\nContent: {body[:1000]}..." # Returning raw content for LLM to summarize
    except Exception as e:
        return f'Error reading email: {str(e)}'

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to the specified recipient."""
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        service.users().messages().send(
            userId='me', body={'raw': raw_message}
        ).execute()
        return f"Email sent successfully to {to}"
    except Exception as e:
        return f"Error sending email: {str(e)}"

# --- Graph Setup ---

class ChatState(TypedDict):
    messages: List[dict]

# Initialize LLMs
llm = init_chat_model(CHAT_MODEL, model_provider='ollama')
tools = [list_unread_emails, summarize_email, send_email]
llm = llm.bind_tools(tools)

def llm_node(state):
    response = llm.invoke(state['messages'])
    return {'messages': state['messages'] + [response]}

def router(state):
    last_message = state['messages'][-1]
    return 'tools' if getattr(last_message, 'tool_calls', None) else 'end'

tool_node = ToolNode(tools)

def tools_node(state):
    result = tool_node.invoke(state)
    return {'messages': state['messages'] + result['messages']}

builder = StateGraph(ChatState)
builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

graph = builder.compile()

# --- Streamlit UI Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(msg['content'])
    elif msg['role'] == 'assistant':
        with st.chat_message("assistant"):
            st.markdown(msg['content'])

# Handle user input
prompt = st.chat_input("How can I help you with your emails?")

if prompt:
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Run the graph
    inputs = {
        # Convert streamlit history to langchain format if needed, 
        # or just pass the full history. LangGraph expects a list of messages.
        # We need to be careful not to duplicate.
        # Simplest: Build a fresh list for the graph from session state
        'messages': [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages
        ]
    }
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # We invoke the graph. Since it's stateful, we get the final state.
        # Streaming is better but invocation is easier for now.
        final_state = graph.invoke(inputs)
        
        # The graph returns the full history. We just want the new messages.
        # Actually, let's just grab the last message content.
        # But wait, the graph might have multiple steps (LLM -> Tool -> LLM).
        # We want the final response from the assistant.
        
        last_message = final_state['messages'][-1]
        full_response = last_message.content
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
