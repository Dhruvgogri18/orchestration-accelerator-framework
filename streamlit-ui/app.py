import json
import re
import threading

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx

from database import (
    add_message,
    create_chat,
    delete_chat,
    get_chat,
    get_chats,
    get_messages,
    initialize_database,
    update_chat_run_state,
    update_chat_title,
)
from workflow import (
    USE_MOCK,
    OrchestrationApiError,
    PreDeployCheckFailed,
    answer_enrichment,
    api_get_session,
    api_get_status,
    approve_plan,
    create_session,
    deploy_workflow,
    fill_spec_placeholders,
    generate_asl,
    reject_plan,
    revalidate_asl,
    send_message,
)

st.set_page_config(
    page_title="Workflow Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()

st.markdown(
    """
<style>
/* ===== Design tokens — single source of truth for the palette =====
   Neutral/graphite theme: no saturated accent hue at all. Primary actions
   (New Chat) use solid near-black graphite. Active chat in history uses a
   soft brand tint (not a solid fill) so selection is clear without looking
   like another CTA. Approve/Reject use green/red outlines. User bubbles are
   soft light-gray — the pattern most production chat UIs (ChatGPT, Claude)
   use so chrome doesn't compete with the conversation. Semantic
   red/amber/green from Streamlit (error/warning/success) stay the only
   strong status colors.
   Since this file is self-contained (no .streamlit/config.toml), every
   widget that would normally inherit theme.primaryColor is overridden
   below to use --brand instead of Streamlit's default red. */
:root {
  --brand: #1f2937;
  --brand-hover: #111827;
  --brand-tint: rgba(31, 41, 55, 0.06);
  --brand-ring: rgba(31, 41, 55, 0.18);
  --text-muted: #6b7280;
  --border: #d8dee9;
  --surface: #ffffff;
  --popover-surface: #f2f3f5;
  --bubble-bg: #f1f2f4;
  --bubble-text: #111827;
}

/* ---- Repoint every Streamlit "primary" widget onto --brand ----
   Covers both current and older Streamlit DOM (kind= attr vs testid),
   since there's no config.toml primaryColor to inherit from. */
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {
  background-color: var(--brand) !important;
  border-color: var(--brand) !important;
  color: #fff !important;
}
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background-color: var(--brand-hover) !important;
  border-color: var(--brand-hover) !important;
}
button[kind="primary"]:focus-visible,
[data-testid="baseButton-primary"]:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible {
  box-shadow: 0 0 0 0.2rem var(--brand-ring) !important;
}

/* ---- Secondary = outline action (Reject / request changes) ----
   Visible border + light surface so it reads as a real button, not
   a text link. Primary stays solid graphite for Approve. */
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"] {
  color: var(--bubble-text) !important;
  background-color: #ffffff !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
button[kind="secondary"]:hover,
button[kind="secondary"]:focus,
button[kind="secondary"]:focus-visible,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-secondary"]:focus,
[data-testid="baseButton-secondary"]:focus-visible,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:focus,
[data-testid="stBaseButton-secondary"]:focus-visible {
  background-color: var(--brand-tint) !important;
  border-color: var(--brand) !important;
  color: var(--brand) !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Tertiary stays quiet (sidebar chat titles, etc.) */
button[kind="tertiary"],
button[kind="tertiary"]:hover,
button[kind="tertiary"]:focus,
button[kind="tertiary"]:focus-visible,
[data-testid="baseButton-tertiary"],
[data-testid="baseButton-tertiary"]:hover,
[data-testid="baseButton-tertiary"]:focus,
[data-testid="baseButton-tertiary"]:focus-visible,
[data-testid="stBaseButton-tertiary"],
[data-testid="stBaseButton-tertiary"]:hover,
[data-testid="stBaseButton-tertiary"]:focus,
[data-testid="stBaseButton-tertiary"]:focus-visible {
  color: var(--bubble-text) !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
button[kind="tertiary"]:hover,
button[kind="tertiary"]:focus,
button[kind="tertiary"]:focus-visible,
[data-testid="baseButton-tertiary"]:hover,
[data-testid="baseButton-tertiary"]:focus,
[data-testid="baseButton-tertiary"]:focus-visible,
[data-testid="stBaseButton-tertiary"]:hover,
[data-testid="stBaseButton-tertiary"]:focus,
[data-testid="stBaseButton-tertiary"]:focus-visible {
  background-color: var(--brand-tint) !important;
  color: var(--brand) !important;
  outline: none !important;
}

/* Text inputs / text areas: neutral theme borders (kill Streamlit red
   invalid / required / focus rings on placeholders + feedback). */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input:hover,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stTextInput"] input:invalid,
[data-testid="stTextInput"] input:required,
[data-testid="stTextInput"] input:-moz-ui-invalid,
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] textarea:hover,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] textarea:focus-visible,
[data-testid="stTextArea"] textarea:invalid,
[data-testid="stTextArea"] textarea:required,
[data-testid="stTextArea"] textarea:-moz-ui-invalid {
  border-color: var(--border) !important;
  box-shadow: none !important;
  outline: none !important;
  caret-color: var(--brand) !important;
}
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"]:hover,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] div[data-baseweb="base-input"],
[data-testid="stTextArea"] div[data-baseweb="base-input"]:hover,
[data-testid="stTextArea"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextInputRootElement"],
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="base-input"],
[data-testid="stTextArea"] [data-baseweb="textarea"] {
  border-color: var(--border) !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] textarea:focus-visible,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 1px var(--brand) !important;
}

/* ---------------- Tabs ---------------- */

button[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--brand) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--brand) !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: var(--brand) !important;
}

[data-baseweb="tab-list"] {
    position: relative !important;
    border-bottom: none !important;
}

/* Grey baseline */
[data-baseweb="tab-list"]::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 1px;
    background: var(--border);
}

/* Moving underline */
[data-baseweb="tab-highlight"] {
    background: var(--brand) !important;
    height: 2px !important;          /* same visual weight as Streamlit */
    border-radius: 999px !important;
    bottom: 0 !important;
    z-index: 5 !important;
}

/* Plan-review action buttons — scoped per column so styles don't leak
   across the shared horizontal row.
   Approve: green outline (white bg). Reject: red outline (white bg). */
.plan-actions-marker { display: none; }
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="stButton"] button,
[data-testid="stColumn"]:has(.approve-btn-marker) button[kind="primary"],
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="baseButton-primary"],
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="stBaseButton-primary"] {
  background-color: #ffffff !important;
  border: 1px solid #16a34a !important;
  color: #15803d !important;
  box-shadow: none !important;
}
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="stButton"] button:hover,
[data-testid="stColumn"]:has(.approve-btn-marker) button[kind="primary"]:hover,
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="baseButton-primary"]:hover,
[data-testid="stColumn"]:has(.approve-btn-marker) [data-testid="stBaseButton-primary"]:hover {
  background-color: #f0fdf4 !important;
  border-color: #15803d !important;
  color: #166534 !important;
}
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="stButton"] button,
[data-testid="stColumn"]:has(.reject-btn-marker) button[kind="secondary"],
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="baseButton-secondary"],
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="stBaseButton-secondary"] {
  background-color: #ffffff !important;
  border: 1px solid #dc2626 !important;
  color: #dc2626 !important;
  box-shadow: none !important;
}
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="stButton"] button:hover,
[data-testid="stColumn"]:has(.reject-btn-marker) button[kind="secondary"]:hover,
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="baseButton-secondary"]:hover,
[data-testid="stColumn"]:has(.reject-btn-marker) [data-testid="stBaseButton-secondary"]:hover {
  background-color: #fef2f2 !important;
  border-color: #b91c1c !important;
  color: #b91c1c !important;
}

/* The visible select shell */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
  min-height: 2rem !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  color: var(--bubble-text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Hover only */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
  border-color: var(--brand) !important;
  box-shadow: none !important;
  outline: none !important;
}

/* When active/open/focused, keep ONE clean border only */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"][aria-expanded="true"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div[aria-expanded="true"] {
  border: 1px solid var(--brand) !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Kill all inner borders, inset shadows, top/bottom rules */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *:focus,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *:focus-visible {
  box-shadow: none !important;
  outline: none !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div::before,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div::after,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *::before,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div *::after {
  content: none !important;
  display: none !important;
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* Text and arrow */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
  color: var(--bubble-text) !important;
  fill: var(--bubble-text) !important;
  stroke: var(--bubble-text) !important;
}

/* Hide caret and any typing feel */
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
  caret-color: transparent !important;
}

/* Popup menu */
div[data-baseweb="popover"] > div,
div[role="listbox"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
  outline: none !important;
}

div[role="option"] {
  background: var(--surface) !important;
  color: var(--bubble-text) !important;
}

div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
  background: var(--brand-tint) !important;
  color: var(--brand) !important;
}

/* ===== Chat options popover =====
   Streamlit renders st.popover in a BaseWeb portal. The real outer
   container is stPopoverBody, so do not scope this under stSidebar. */

[data-testid="stPopoverBody"] {
  width: max-content !important;
  min-width: 280px !important;
  max-width: 340px !important;
  padding: 0.85rem !important;
  background: var(--popover-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12) !important;
}

/* Override the earlier generic:
   div[data-baseweb="popover"] > div { background: var(--surface) }
   This selector has greater specificity and removes the white inner layer. */
[data-testid="stPopoverBody"][data-baseweb="popover"] > div,
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
[data-testid="stPopoverBody"] [data-testid="stElementContainer"],
[data-testid="stPopoverBody"] [data-testid="stMarkdown"],
[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"],
[data-testid="stPopoverBody"] [data-testid="stLayoutWrapper"],
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"],
[data-testid="stPopoverBody"] [data-testid="stColumn"] {
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

[data-testid="stPopoverBody"] .chat-options-heading {
  margin: 0 0 0.6rem !important;
  color: var(--bubble-text) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
}

/* Rename input: permanent black border, including after blur */
[data-testid="stPopoverBody"] [data-testid="stTextInput"] {
  position: relative !important;
  margin-bottom: 0.75rem !important;
}

[data-testid="stPopoverBody"] [data-testid="stTextInputRootElement"],
[data-testid="stPopoverBody"] div[data-baseweb="base-input"],
[data-testid="stPopoverBody"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stPopoverBody"] div[data-baseweb="base-input"]:focus-within {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}

[data-testid="stPopoverBody"] [data-testid="stTextInput"] input,
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input:hover,
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input:focus,
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input:focus-visible {
  height: 2.5rem !important;
  padding: 0 0.85rem !important;
  background: var(--popover-surface) !important;
  border: 1px solid #111827 !important;
  border-radius: 10px !important;
  color: var(--bubble-text) !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Button row */
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"] {
  gap: 0.5rem !important;
}

[data-testid="stPopoverBody"] [data-testid="stButton"] button {
  height: 2.4rem !important;
  min-height: 2.4rem !important;
  padding: 0 0.6rem !important;
  background: var(--popover-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--bubble-text) !important;
  box-shadow: none !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  white-space: nowrap !important;
}

[data-testid="stPopoverBody"] [data-testid="stButton"] button:hover,
[data-testid="stPopoverBody"] [data-testid="stButton"] button:focus-visible {
  background: var(--popover-surface) !important;
  border-color: var(--brand) !important;
  color: var(--brand) !important;
  box-shadow: none !important;
}

/* The destructive action is the second button column */
[data-testid="stPopoverBody"]
[data-testid="stHorizontalBlock"]
[data-testid="stColumn"]:nth-of-type(2)
[data-testid="stButton"] button {
  color: #dc2626 !important;
}

[data-testid="stPopoverBody"]
[data-testid="stHorizontalBlock"]
[data-testid="stColumn"]:nth-of-type(2)
[data-testid="stButton"] button:hover,
[data-testid="stPopoverBody"]
[data-testid="stHorizontalBlock"]
[data-testid="stColumn"]:nth-of-type(2)
[data-testid="stButton"] button:focus-visible {
  background: var(--popover-surface) !important;
  border-color: #dc2626 !important;
  color: #dc2626 !important;
}

/* Hide Streamlit's “Press Enter to apply” hint */
[data-testid="stPopoverBody"] [data-testid="InputInstructions"],
[data-testid="stPopoverBody"] [data-testid="stTextInput"] [data-testid="InputInstructions"] {
  display: none !important;
}

/* Links */
a, a:visited {
  color: var(--brand);
}

/* Main chat area */
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 920px !important;
  padding-top: 1rem !important;
  padding-bottom: 8rem !important;
}

/* Do not reserve layout space for helper JavaScript iframes */
[data-testid="stMainBlockContainer"] [data-testid="stIFrame"][height="0"],
.block-container [data-testid="stIFrame"][height="0"] {
  display: block !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

[data-testid="stMainBlockContainer"]
[data-testid="stElementContainer"]:has(> [data-testid="stIFrame"][height="0"]),
.block-container
[data-testid="stElementContainer"]:has(> [data-testid="stIFrame"][height="0"]) {
  display: none !important;
}

/* Fixed sidebar width while open */
[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 21rem !important;
  max-width: 21rem !important;
  width: 21rem !important;
  flex: 0 0 21rem !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
  width: 21rem !important;
  min-width: 21rem !important;
  max-width: 21rem !important;
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
}

/* Sidebar collapsed state */
[data-testid="stSidebar"][aria-expanded="false"] {
  min-width: 0 !important;
  max-width: 0 !important;
  width: 0 !important;
  flex: 0 0 0 !important;
}
[data-testid="stSidebarResizeHandle"] {
  display: none !important;
  width: 0 !important;
  pointer-events: none !important;
}

/* Lock the outer sidebar; history is the sole scrollable region */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  display: flex !important;
  flex-direction: column !important;
  height: 100vh !important;
  overflow: hidden !important;
}

/* Compact fixed sidebar controls */
[data-testid="stSidebar"] h1 {
  font-size: 1.55rem !important;
  margin-top: 0 !important;
  margin-bottom: 0.2rem !important;
  padding-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  font-size: 0.72rem !important;
  margin-bottom: -0.35rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
  margin-bottom: -0.7rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
  font-size: 0.78rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
  min-height: 2rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
  min-height: 2rem !important;
  padding-top: 0.25rem !important;
  padding-bottom: 0.25rem !important;
}
[data-testid="stSidebar"] hr {
  margin: 0.9rem 0 !important;
}
.sidebar-heading {
  margin: 0.1rem 0 0.3rem !important;
  font-size: 0.68rem !important;
  color: var(--text-muted);
  font-weight: 600;
  letter-spacing: 0.06em;
}

/* Only the chat history st.container scrolls */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker) {
    height: calc(100vh - 255px) !important;
    max-height: calc(100vh - 255px) !important;
    min-height: 120px !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;

    padding-right: 0.55rem !important;
    padding-bottom: 4rem !important;
    margin-bottom: 0 !important;
    box-sizing: border-box !important;
}

/* Fixed chat input */
[data-testid="stChatInput"] {
  position: fixed;
  bottom: 1rem;
  left: calc(50% + 8.5rem);
  transform: translateX(-50%);
  width: min(820px, calc(100vw - 23rem));
  z-index: 999;
  padding: 0.55rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 8px 25px rgba(15, 23, 42, 0.16);
}

[data-testid="stChatInput"]:focus-within {
  border-color: var(--brand) !important;
  box-shadow:
    0 8px 25px rgba(15, 23, 42, 0.16),
    0 0 0 1px var(--brand) !important;
}

[data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stChatInput"] {
  left: 50%;
  width: min(820px, calc(100vw - 3rem));
}

[data-testid="stChatInput"] textarea {
  font-size: 1rem;
  background: transparent !important;
  max-height: 12rem;
  overflow-y: auto;
  resize: none;
}

/* Empty state: undo leftover auto-grow height without forcing a tiny box. */
[data-testid="stChatInput"] textarea:placeholder-shown {
  height: auto !important;
  min-height: 1.75rem !important;
  max-height: none !important;
  overflow-y: hidden !important;
}



/* Remove the INNER rounded red border completely */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div:focus-within,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}

/* Kill textarea/browser invalid/focus styling */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] textarea:invalid,
[data-testid="stChatInput"] textarea:required,
[data-testid="stChatInput"] textarea:-moz-ui-invalid {
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}

/* Graphviz diagram spacing */
[data-testid="stGraphVizChart"] {
    margin-top: 0.75rem !important;
}

/* Shared avatar base */
.chat-avatar {
  flex-shrink: 0;
  width: 1.85rem;
  height: 1.85rem;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 0.15rem;
}
.chat-avatar svg {
  width: 1rem;
  height: 1rem;
  display: block;
}

/* User messages — right-aligned: bubble then avatar */
.user-message-row {
  display: flex;
  justify-content: flex-end;
  align-items: flex-start;
  gap: 0.65rem;
  margin: 0.6rem 0 1rem;
}
.user-message-bubble {
  max-width: 74%;
  padding: 0.75rem 1rem;
  border-radius: 18px 18px 4px 18px;
  background: var(--bubble-bg);
  color: var(--bubble-text);
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.user-avatar {
  background: #e5e7eb;
  color: #374151;
}

/* Assistant messages — left-aligned: avatar then bubble */
.assistant-message-row {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 0.65rem;
  margin: 0.6rem 0 1.25rem;
  max-width: 86%;
}
.assistant-avatar {
  background: var(--brand);
  color: #fff;
}
.assistant-message-bubble {
  flex: 1;
  min-width: 0;
  padding: 0.75rem 1rem;
  border-radius: 4px 18px 18px 18px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  color: var(--bubble-text);
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.assistant-response {
  margin: 0.75rem 0 1.6rem;
  max-width: 88%;
}
.assistant-response-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.55rem;
  color: var(--text-muted);
  font-size: 0.8rem;
  font-weight: 500;
}
.assistant-response-label .chat-avatar {
  width: 1.4rem;
  height: 1.4rem;
  margin-top: 0;
}
.assistant-response-label .chat-avatar svg {
  width: 0.8rem;
  height: 0.8rem;
}

/* Typing indicator — modern chatbot “assistant is responding” row */
.typing-indicator-row {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 0.65rem;
  margin: 0.6rem 0 1.25rem;
  max-width: 86%;
}
.typing-indicator-bubble {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.7rem 1rem;
  border-radius: 4px 18px 18px 18px;
  background: #ffffff;
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.9rem;
  line-height: 1.4;
}
.typing-dots {
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
}
.typing-dots span {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 999px;
  background: var(--brand);
  opacity: 0.35;
  animation: typing-bounce 1.2s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes typing-bounce {
  0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
  30% { opacity: 1; transform: translateY(-3px); }
}
/* Hide Streamlit’s default spinner chrome — we use the chat typing row. */
div[data-testid="stSpinner"] {
  display: none !important;
}

/* Stage progress — auto-advancing checklist shown instead of a plain
   "Thinking…" spinner while a backend request is in flight. Same bubble
   shape as an assistant message so it reads as part of the conversation. */
.stage-progress-row {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 0.65rem;
  margin: 0.6rem 0 1.25rem;
  max-width: 86%;
}
.stage-progress-bubble {
  flex: 1;
  min-width: 0;
  padding: 0.8rem 1rem;
  border-radius: 4px 18px 18px 18px;
  background: #ffffff;
  border: 1px solid var(--border);
}
.stage-progress-header {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-bottom: 0.55rem;
  letter-spacing: 0.01em;
}
.stage-progress-list {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.stage-item {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  font-size: 0.88rem;
  line-height: 1.3;
  color: var(--text-muted);
}
.stage-item .stage-icon {
  width: 1.05rem;
  height: 1.05rem;
  flex: none;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}
.stage-item.done {
  color: var(--bubble-text);
}
.stage-item.done .stage-icon {
  color: #16a34a;
}
.stage-item.active {
  color: var(--bubble-text);
  font-weight: 600;
}
.stage-item .stage-dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 999px;
  background: var(--border);
  display: block;
}
.stage-item.active .stage-dot {
  background: var(--brand);
  animation: stage-pulse 1.1s ease-in-out infinite;
}
@keyframes stage-pulse {
  0%, 100% { opacity: 0.45; transform: scale(0.85); }
  50% { opacity: 1; transform: scale(1.2); }
}

/* ===== Compact chat history rows ===== */

/* Zero out Streamlit’s default vertical gap inside the history scroller */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="element-container"] {
  margin: 0 !important;
  padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stHorizontalBlock"] {
  margin: 0 0 0.2rem 0 !important;
  padding: 0 !important;
  gap: 0.25rem !important;
  min-height: 2rem !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* Flexbox items default to min-width:auto, which blocks ellipsis
   truncation and lets the label wrap onto a second line instead. Force
   every wrapper between the row and the button to allow shrinking. */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stHorizontalBlock"] > * {
  min-width: 0 !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stColumn"] {
  min-width: 0 !important;
  flex-shrink: 1 !important;
  overflow: hidden !important;
}

/* Keep the three-dot menu column compact and slightly inset */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
  flex: 0 0 auto !important;
  padding-right: 0.1rem !important;
}

/* Title button: fixed height, never wrap */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stButton"] {
  min-width: 0 !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stButton"] button {
  height: 2rem !important;
  min-height: 2rem !important;
  max-height: 2rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  padding-left: 0.6rem !important;
  padding-right: 0.6rem !important;
  overflow: hidden !important;
  white-space: nowrap !important;
  text-overflow: ellipsis !important;
  display: block !important;
  line-height: 2rem !important;
  text-align: left !important;
}

/* Force every text node inside the button onto one line */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stButton"] button p,
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stButton"] button span,
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stButton"] button div {
  margin: 0 !important;
  padding: 0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  line-height: 2rem !important;
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  text-align: left !important;
  height: 2rem !important;
}

/* Active chat title — soft tint (not solid primary). New Chat stays solid
   graphite; history selection should read as “here”, not as another CTA. */
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[kind="primary"],
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[data-testid="baseButton-primary"],
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] [data-testid="stBaseButton-primary"] {
    display: block !important;
    height: 2rem !important;
    min-height: 2rem !important;
    max-height: 2rem !important;
    line-height: 2rem !important;
    padding: 0 0.6rem !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    text-align: left !important;
    background-color: var(--brand-tint) !important;
    border: 1px solid transparent !important;
    border-left: 3px solid var(--brand) !important;
    color: var(--brand) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover,
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] [data-testid="stBaseButton-primary"]:hover {
    background-color: rgba(31, 41, 55, 0.10) !important;
    color: var(--brand-hover) !important;
    border-color: transparent !important;
}

[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[kind="primary"] *,
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[data-testid="baseButton-primary"] *,
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] [data-testid="stBaseButton-primary"] * {
    display: block !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    line-height: 2rem !important;
    color: inherit !important;
}

/* ⋮ menu stays compact and aligned */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  min-width: 2rem !important;
  width: 2rem !important;
  margin: 0 !important;
  padding: 0 !important;
  box-sizing: border-box !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] button {
  min-height: 2rem !important;
  height: 2rem !important;
  min-width: 2rem !important;
  width: 2rem !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
  gap: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] button svg {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] button:hover {
  background: var(--brand-tint) !important;
  border-radius: 6px !important;
  color: var(--brand) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] button p {
  margin: 0 !important;
  text-align: center !important;
}

/* Three-dot chat menu — only ⋮, hide dropdown caret (fallback outside history) */
[data-testid="stSidebar"] [data-testid="stPopover"] button {
  min-height: 2rem !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
  gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button svg {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button p {
  margin: 0 !important;
  text-align: center !important;
}

/* Neutralize Streamlit's default inline-code color (session id, etc.)
   so it reads as metadata, not a third accent color */
.stCaption code, [data-testid="stCaptionContainer"] code {
  color: var(--text-muted) !important;
  background: var(--secondary-background-color, #f3f4f6) !important;
}

@media (max-width: 900px) {
  [data-testid="stChatInput"] {
    left: 50%;
    width: calc(100vw - 2rem);
  }
  .user-message-bubble {
    max-width: 90%;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLE_PROMPTS = [
    "Validate an S3 file, run a Glue job, then send a Slack notification",
    "On S3 upload, validate file type, process with Lambda, notify on failure",
    "Scheduled nightly ETL with success and failure notifications",
]

components.html(
    """
    <script>
    (function() {
      const doc = window.parent.document;

      function disableSpellcheck() {
        const root = doc.querySelector('[data-testid="stChatInput"]');
        const ta = root && root.querySelector('textarea');
        if (!ta) return;
        if (ta.getAttribute('spellcheck') !== 'false') {
          ta.setAttribute('spellcheck', 'false');
          ta.setAttribute('autocomplete', 'off');
          ta.setAttribute('autocorrect', 'off');
          ta.setAttribute('autocapitalize', 'off');
        }
        if (ta.hasAttribute('required')) {
          ta.removeAttribute('required');
        }
        if (typeof ta.setCustomValidity === 'function') {
          ta.setCustomValidity('');
        }
      }

      function truncateHistoryTitles() {
        doc.querySelectorAll(
          '[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]'
        ).forEach((wrap) => {
          if (!wrap.querySelector('.sidebar-history-marker')) return;
          wrap.querySelectorAll('[data-testid="stButton"] button').forEach((btn) => {
            const apply = (el) => {
              el.style.setProperty('white-space', 'nowrap', 'important');
              el.style.setProperty('overflow', 'hidden', 'important');
              el.style.setProperty('text-overflow', 'ellipsis', 'important');
              el.style.setProperty('min-width', '0', 'important');
              el.style.setProperty('max-width', '100%', 'important');
              el.style.setProperty('display', 'block', 'important');
            };
            apply(btn);
            btn.querySelectorAll('*').forEach(apply);
          });
        });
      }

      function applyAll() {
        disableSpellcheck();
        truncateHistoryTitles();
      }

      applyAll();
      if (!window.__wfSidebarObserver) {
        window.__wfSidebarObserver = new MutationObserver(applyAll);
        window.__wfSidebarObserver.observe(doc.body, { childList: true, subtree: true });
      }
    })();
    </script>
    """,
    height=0,
)

components.html(
    """
    <script>
    (function () {
      const doc = window.parent.document;

      function wireSelectBlur() {
        const selects = doc.querySelectorAll('[data-testid="stSidebar"] [data-baseweb="select"]');

        selects.forEach((select) => {
          if (select.dataset.blurWired === "1") return;
          select.dataset.blurWired = "1";

          select.addEventListener("mousedown", () => {
            requestAnimationFrame(() => {
              const expanded =
                select.getAttribute("aria-expanded") === "true" ||
                select.querySelector('[aria-expanded="true"]');

              if (!expanded) {
                const input = select.querySelector("input");
                if (input) input.blur();
              }
            });
          });

          select.addEventListener("keydown", (e) => {
            if (e.key === "Escape" || e.key === "Tab") {
              const input = select.querySelector("input");
              if (input) input.blur();
            }
          });
        });
      }

      wireSelectBlur();
      const observer = new MutationObserver(wireSelectBlur);
      observer.observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    height=0,
)

STAGE_PREFIX = {
    "clarifying": "… ",
    "plan_review": "Plan ",
    "spec_approved": "✓ ",
    "out_of_scope": "⊘ ",
    "failed": "! ",
}


def create_welcome_message(chat_id):
    mode = "mock mode" if USE_MOCK else "connected to orchestration API"
    add_message(
        chat_id,
        "assistant",
        (
            "Welcome to the Workflow Assistant. Describe an AWS Step Functions "
            "orchestration in natural language (steps, services, success/failure "
            "behavior). I may ask a clarifying question, then show a plan "
            "(summary + diagram + placeholders). You can approve the plan or "
            f"request changes. ({mode}.)\n\n"
            "I only help design Step Functions workflows — not general questions. "
            "After you approve a plan, enrichment produces a workflow spec, then "
            "ASL is generated, validated (including live AWS validation when "
            "credentials are configured), and shown as a console-style graph."
        ),
    )


def set_active_chat(chat_id, focus: bool = True) -> None:
    """Switch the open chat and persist it in the URL so a browser refresh
    stays on the same conversation instead of opening a new draft.

    Does not clear other chats' in-flight work — pending prompts stay
    queued and resume when the user returns to that chat.
    """
    prev = st.session_state.get("active_chat_id")
    st.session_state.active_chat_id = chat_id
    # Force the next paint to treat the thread as a new widget tree so
    # Streamlit does not reuse elements (and visible content) from the
    # chat we just left.
    if prev != chat_id:
        st.session_state["_thread_nonce"] = (
            st.session_state.get("_thread_nonce", 0) + 1
        )
    if focus:
        st.session_state.focus_chat_input = True
    try:
        st.query_params["chat"] = str(chat_id)
    except Exception:
        pass


def _chat_id_from_query_params():
    """Read ?chat=<id> from the URL. Returns int or None."""
    try:
        raw = st.query_params.get("chat")
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def ensure_active_chat():
    """Keep the in-session active chat if it still exists; otherwise open a
    draft. Prefer the chat id from the URL (refresh / deep-link), then an
    unused draft, then create a new one."""
    active_id = st.session_state.get("active_chat_id")
    if active_id is not None and get_chat(active_id) is not None:
        # Keep URL in sync when session state already has a valid chat.
        try:
            if st.query_params.get("chat") != str(active_id):
                st.query_params["chat"] = str(active_id)
        except Exception:
            pass
        return

    url_id = _chat_id_from_query_params()
    if url_id is not None and get_chat(url_id) is not None:
        set_active_chat(url_id)
        return

    # Prefer reusing an existing unused draft so we don't spam empty chats.
    for chat in get_chats():
        if is_draft_chat(chat["id"]):
            set_active_chat(chat["id"])
            return

    chat_id = create_chat()
    create_welcome_message(chat_id)
    set_active_chat(chat_id)


def is_draft_chat(chat_id):
    if get_chat(chat_id) is None:
        return False
    messages = get_messages(chat_id)
    return not any(m["role"] == "user" for m in messages)


def get_active_session_id(chat_id):
    # Freshly created session, not yet persisted to chat["run_id"] (that
    # only happens once the full response comes back) — see
    # process_user_prompt(), which stashes it here immediately so the
    # live-status checklist has something to poll from the very first
    # message of a new chat, not just from the second message onward.
    hints = st.session_state.get("_pending_session_ids")
    if isinstance(hints, dict) and hints.get(chat_id):
        return hints[chat_id]
    chat = get_chat(chat_id)
    if chat and chat.get("run_id"):
        return chat["run_id"]
    for msg in reversed(get_messages(chat_id)):
        art = msg.get("artifact") or {}
        sid = art.get("session_id") or art.get("run_id")
        if sid:
            return sid
    return None


def _remember_session_id(chat_id, session_id) -> None:
    """Session-state hint so get_active_session_id() finds a just-created
    session immediately. Without this, a brand-new chat's first message has
    no session_id to poll GET /status with until the whole request
    finishes, so the stage checklist has nothing to advance on and sits on
    step one the entire time.
    """
    hints = st.session_state.get("_pending_session_ids")
    if not isinstance(hints, dict):
        hints = {}
    hints[chat_id] = session_id
    st.session_state._pending_session_ids = hints


def create_new_chat():
    """Open a blank draft for a new conversation.

    Reuse the newest empty draft (highest id, no user messages) so
    New Chat → switch to old chat → New Chat again stays on the same
    draft (e.g. 40) instead of creating 41. Never jump backward to an
    older leftover draft (e.g. from 39 down to 38).
    """
    active_id = st.session_state.get("active_chat_id")
    if active_id and get_chat(active_id) is not None and is_draft_chat(active_id):
        set_active_chat(active_id)
        return

    try:
        active_n = int(active_id) if active_id is not None else 0
    except (TypeError, ValueError):
        active_n = 0

    drafts = [c for c in get_chats() if is_draft_chat(c["id"])]
    if drafts:
        newest = max(drafts, key=lambda c: int(c["id"]))
        # Only reuse if this draft is at/after the current chat id —
        # avoids selecting an older empty row (38) while on chat 39.
        if int(newest["id"]) >= active_n:
            set_active_chat(newest["id"])
            return

    chat_id = create_chat()
    create_welcome_message(chat_id)
    set_active_chat(chat_id)


def create_chat_title(prompt):
    title = " ".join(prompt.strip().split())
    if not title:
        return "Untitled workflow"
    if len(title) > 45:
        return title[:45].rstrip() + "..."
    return title


def delete_chat_and_select_next(chat_id):
    delete_chat(chat_id)
    clear_in_flight(chat_id)
    chats = get_chats()
    if chats:
        set_active_chat(chats[0]["id"])
    else:
        create_new_chat()

def close_chat_options_popover():
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          function closePopovers() {
            doc.querySelectorAll('[data-testid="stPopoverBody"]').forEach((body) => {
              body.remove();
            });

            doc.querySelectorAll('[data-testid="stPopover"] button').forEach((button) => {
              button.setAttribute("aria-expanded", "false");
            });
          }

          closePopovers();
          requestAnimationFrame(closePopovers);
          setTimeout(closePopovers, 50);
        })();
        </script>
        """,
        height=0,
    )

def render_sidebar_chat(chat):
    chat_id = chat["id"]
    is_active = chat_id == st.session_state.active_chat_id

    full_title = chat["title"]
    limit = 26 if is_active else 32
    label = full_title
    if len(label) > limit:
        label = label[:limit].rstrip() + "…"

    chat_column, option_column = st.columns(
        [0.90, 0.10],
        gap="small",
        vertical_alignment="center",
    )

    with chat_column:
        if st.button(
            label,
            key=f"open_chat_{chat_id}",
            use_container_width=True,
            type="primary" if is_active else "tertiary",
        ):
            set_active_chat(chat_id)
            st.rerun()

    with option_column:
      nonce_key = f"popover_nonce_{chat_id}"
      nonce = st.session_state.get(nonce_key, 0)

      with st.container(key=f"chat_options_wrap_{chat_id}_{nonce}"):
          with st.popover("⋮", use_container_width=False):
              confirm_key = f"confirm_delete_{chat_id}"
              confirming = st.session_state.get(confirm_key, False)

              st.markdown(
                  '<div class="chat-options-marker chat-options-heading">'
                  "Chat options</div>",
                  unsafe_allow_html=True,
              )

              if not confirming:
                  new_title = st.text_input(
                      "Rename chat",
                      value=chat["title"],
                      key=f"rename_input_{chat_id}",
                      label_visibility="collapsed",
                  )

                  c1, c2 = st.columns(2)

                  with c1:
                      if st.button(
                          "Save",
                          key=f"save_rename_{chat_id}",
                          use_container_width=True,
                      ):
                          update_chat_title(chat_id, new_title)
                          st.session_state[nonce_key] = nonce + 1
                          close_chat_options_popover()
                          st.rerun()

                  with c2:
                      if st.button(
                          "Delete",
                          key=f"delete_chat_{chat_id}",
                          use_container_width=True,
                      ):
                          st.session_state[confirm_key] = True
                          st.rerun()

              else:
                  st.markdown(
                      '<div class="chat-options-marker chat-options-confirm-text">'
                      "Delete this chat? This cannot be undone.</div>",
                      unsafe_allow_html=True,
                  )

                  c1, c2 = st.columns(2)

                  with c1:
                      if st.button(
                          "Cancel",
                          key=f"cancel_delete_{chat_id}",
                          use_container_width=True,
                      ):
                          st.session_state[confirm_key] = False
                          st.rerun()

                  with c2:
                      if st.button(
                          "Confirm",
                          key=f"confirm_delete_button_{chat_id}",
                          use_container_width=True,
                      ):
                          st.session_state[confirm_key] = False
                          st.session_state[nonce_key] = nonce + 1
                          close_chat_options_popover()
                          delete_chat_and_select_next(chat_id)
                          st.rerun()


def _escape_html(content: str) -> str:
    return (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# Inline SVG icons for chat avatars (no external assets).
# Robot assistant: head, antenna, eyes, mouth, side arms.
_ASSISTANT_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="5" y="9" width="14" height="10" rx="2"/>'
    '<path d="M12 3v3"/>'
    '<circle cx="12" cy="3" r="1.5" fill="currentColor" stroke="none"/>'
    '<circle cx="9" cy="13" r="1.2" fill="currentColor" stroke="none"/>'
    '<circle cx="15" cy="13" r="1.2" fill="currentColor" stroke="none"/>'
    '<path d="M9 17h6"/>'
    '<path d="M5 12H3"/>'
    '<path d="M21 12h-2"/>'
    "</svg>"
)

_USER_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="8" r="4"/>'
    '<path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>'
    "</svg>"
)


def render_user_message(content):
    safe = _escape_html(content)
    st.markdown(
        (
            '<div class="user-message-row">'
            f'<div class="user-message-bubble">{safe}</div>'
            f'<div class="chat-avatar user-avatar">{_USER_ICON}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_assistant_message(content):
    """Plain assistant chat bubble (left side) — for clarifying / text replies."""
    safe = _escape_html(content)
    st.markdown(
        (
            '<div class="assistant-message-row">'
            f'<div class="chat-avatar assistant-avatar">{_ASSISTANT_ICON}</div>'
            f'<div class="assistant-message-bubble">{safe}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_typing_indicator(label: str = "Thinking…"):
    """Chat-native loading row (avatar + dots + short label)."""
    safe = _escape_html(label)
    st.markdown(
        (
            '<div class="typing-indicator-row">'
            f'<div class="chat-avatar assistant-avatar">{_ASSISTANT_ICON}</div>'
            '<div class="typing-indicator-bubble">'
            '<span class="typing-dots" aria-hidden="true">'
            "<span></span><span></span><span></span>"
            "</span>"
            f"<span>{safe}</span>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Stage progress — auto-advancing checklist of real backend steps
# ---------------------------------------------------------------------------
#
# Ordered (stage_key, display_label) pairs, one list per kind of in-flight
# request. stage_key must match the `stage` string the API reports via
# GET /session/{id}/status (see api/services/llm_trace.py's llm_stage()
# calls, invoked from pipeline_analysis_service.py / enrichment_service.py /
# session_service.py). The list itself is defined here rather than fetched
# from the API so it renders immediately and doesn't jump around — only the
# *current position* moves, driven by the polled stage value.
_EXTRACTION_STEPS: list[tuple[str, str]] = [
    ("pipeline_analysis_env_and_clauses", "Reading through what you described"),
    ("component_identification", "Identifying the workflow's components"),
    ("flow_structure_analysis", "Working out the control flow"),
    ("pipeline_analysis_params_and_integrations", "Extracting parameters & checking integrations"),
]

_APPROVE_STEPS: list[tuple[str, str]] = [
    ("enrichment", "Filling in the remaining details"),
    ("asl_generation", "Generating the Step Functions definition"),
]

_STAGE_CHECK_ICON = (
    '<svg class="stage-icon" viewBox="0 0 20 20" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M4 10.5l3.5 3.5L16 6" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_STAGE_DOT_ICON = '<span class="stage-icon"><span class="stage-dot"></span></span>'


def _stage_steps_for_chat(chat_id) -> list[tuple[str, str]]:
    """Which step checklist applies to whatever request is in flight for
    this chat: approve / answer-enrichment share one (enrichment -> ASL),
    everything else (send message / reject) is the extraction pipeline.
    """
    pending_approve = st.session_state.get("pending_approve")
    if pending_approve and pending_approve.get("chat_id") == chat_id:
        return _APPROVE_STEPS
    pending_enrichment = st.session_state.get("pending_enrichment")
    if pending_enrichment and pending_enrichment.get("chat_id") == chat_id:
        return _APPROVE_STEPS
    return _EXTRACTION_STEPS


def render_stage_progress(chat_id, session_id, fallback_label: str = "Working…"):
    """Checklist that auto-advances as the backend moves through real
    pipeline stages, replacing a plain "Thinking…" spinner. Position comes
    from polling GET /session/{id}/status; if the reported stage isn't one
    we recognize yet (request just started, or a stage name the list
    doesn't know about) the first not-yet-reached step just stays active
    instead of the display breaking or looking stuck on nothing.
    """
    steps = _stage_steps_for_chat(chat_id)
    current_stage = None
    if session_id and not USE_MOCK:
        status = api_get_status(session_id)
        if status.get("busy"):
            current_stage = status.get("stage")

    keys = [key for key, _ in steps]
    current_idx = keys.index(current_stage) if current_stage in keys else 0

    rows = []
    for i, (_, label) in enumerate(steps):
        if i < current_idx:
            state, icon = "done", _STAGE_CHECK_ICON
        elif i == current_idx:
            state, icon = "active", _STAGE_DOT_ICON
        else:
            state, icon = "pending", _STAGE_DOT_ICON
        rows.append(
            f'<div class="stage-item {state}">{icon}<span>{_escape_html(label)}</span></div>'
        )

    st.markdown(
        (
            '<div class="stage-progress-row">'
            f'<div class="chat-avatar assistant-avatar">{_ASSISTANT_ICON}</div>'
            '<div class="stage-progress-bubble">'
            f'<div class="stage-progress-header">{_escape_html(fallback_label)}</div>'
            f'<div class="stage-progress-list">{"".join(rows)}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _bg_jobs() -> dict:
    """Per-session registry of in-flight background API calls, keyed by
    chat_id. Lets the polling fragment below tell whether a job is still
    running without touching the network itself.
    """
    jobs = st.session_state.get("_bg_jobs")
    if not isinstance(jobs, dict):
        jobs = {}
        st.session_state._bg_jobs = jobs
    return jobs


def run_with_live_status(chat_id, work_fn):
    """Run a blocking orchestration-API call on a background thread so the
    main script isn't stuck inside `with st.spinner(...): result = work_fn()`
    for the whole duration — that pattern is why the UI could only ever show
    a static "Thinking…" the entire time. With the call moved to a thread,
    the main script returns immediately and `render_live_status` (below)
    can keep polling GET /session/{id}/status and repainting the label
    while the real work happens.

    First call for a given chat_id starts the thread and returns
    ("running", None, None). Call again (the polling fragment triggers a
    rerun that re-enters this same call site) once the thread finishes to
    get ("done", result, None) or ("error", None, exception) — returned
    exactly once, then the job is cleared.
    """
    jobs = _bg_jobs()
    job = jobs.get(chat_id)

    if job is None:
        state = {"status": "running", "result": None, "error": None}

        def _runner():
            try:
                state["result"] = work_fn()
                state["status"] = "done"
            except Exception as exc:  # noqa: BLE001 -- surfaced to the UI below, not swallowed
                state["error"] = exc
                state["status"] = "error"

        thread = threading.Thread(target=_runner, daemon=True)
        add_script_run_ctx(thread)  # lets the thread touch st.* safely if it ever needs to
        thread.start()
        jobs[chat_id] = state
        return "running", None, None

    if job["status"] == "running":
        return "running", None, None

    jobs.pop(chat_id, None)
    if job["status"] == "error":
        return "error", None, job["error"]
    return "done", job["result"], None


@st.fragment(run_every=1)
def render_live_status(chat_id, session_id, fallback_label="Thinking…"):
    """Polls /session/{id}/status roughly once a second and shows the
    real backend stage ("Working out the control flow…", "Generating
    the Step Functions definition…") instead of a static spinner. Only
    this fragment re-executes on the timer -- the rest of the page (and
    the blocking API call running in its background thread) is untouched.

    Once the background job finishes, triggers one full-page rerun so
    the normal (non-fragment) code path picks up the result and renders
    it exactly as it did before this change.
    """
    job = _bg_jobs().get(chat_id)
    if job is not None and job["status"] != "running":
        st.rerun()
        return

    render_stage_progress(chat_id, session_id, fallback_label=fallback_label)


def typing_label_for_chat(chat_id) -> str:
    """Short, context-aware label while the backend works for THIS chat only."""
    pending_approve = st.session_state.get("pending_approve")
    if pending_approve and pending_approve.get("chat_id") == chat_id:
        return "Approving…"
    pending_enrichment = st.session_state.get("pending_enrichment")
    if pending_enrichment and pending_enrichment.get("chat_id") == chat_id:
        return "Enriching…"
    pending_fill = st.session_state.get("pending_fill_placeholders")
    if pending_fill and pending_fill.get("chat_id") == chat_id:
        return "Applying placeholders…"
    pending_gen = st.session_state.get("pending_generate_asl")
    if pending_gen and pending_gen.get("chat_id") == chat_id:
        return "Generating ASL…"
    pending_reval = st.session_state.get("pending_revalidate_asl")
    if pending_reval and pending_reval.get("chat_id") == chat_id:
        return "Validating ASL…"
    pending_deploy = st.session_state.get("pending_deploy")
    if pending_deploy and pending_deploy.get("chat_id") == chat_id:
        return "Deploying to AWS…"
    chat = get_chat(chat_id)
    stage = (chat or {}).get("stage")
    if stage == "plan_review":
        return "Updating the plan…"
    return "Thinking…"


def render_assistant_block_start():
    """Header for richer assistant cards (plan review, approve, etc.)."""
    st.markdown(
        (
            '<div class="assistant-response-label">'
            f'<div class="chat-avatar assistant-avatar">{_ASSISTANT_ICON}</div>'
            "<span>Assistant</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _pending_prompts() -> dict:
    """Per-chat queued user prompts. Keyed by chat_id so chats stay independent."""
    prompts = st.session_state.get("pending_prompts")
    if not isinstance(prompts, dict):
        prompts = {}
        st.session_state.pending_prompts = prompts
    return prompts


def is_chat_busy(chat_id) -> bool:
    """True while a message, approve, enrichment, ASL, or deploy is in flight for this chat only."""
    if chat_id is None:
        return False
    if chat_id in _pending_prompts():
        return True
    for key in (
        "pending_approve",
        "pending_enrichment",
        "pending_fill_placeholders",
        "pending_generate_asl",
        "pending_revalidate_asl",
        "pending_deploy",
    ):
        pending = st.session_state.get(key)
        if pending and pending.get("chat_id") == chat_id:
            return True
    return False


def is_ui_busy() -> bool:
    """True while any chat has a request in flight (legacy / global checks)."""
    if _pending_prompts():
        return True
    if st.session_state.get("pending_approve"):
        return True
    if st.session_state.get("pending_enrichment"):
        return True
    if st.session_state.get("pending_fill_placeholders"):
        return True
    if st.session_state.get("pending_generate_asl"):
        return True
    if st.session_state.get("pending_revalidate_asl"):
        return True
    return bool(st.session_state.get("pending_deploy"))


def clear_in_flight(chat_id=None):
    """Clear in-flight state. If chat_id is given, only clear that chat."""
    pending_keys = (
        "pending_approve",
        "pending_enrichment",
        "pending_fill_placeholders",
        "pending_generate_asl",
        "pending_revalidate_asl",
        "pending_deploy",
    )
    if chat_id is None:
        st.session_state.pending_prompts = {}
        for k in pending_keys:
            st.session_state[k] = None
        # Legacy keys (older runs / partial upgrades)
        st.session_state.pending_prompt = None
        st.session_state.pending_chat_id = None
        return

    prompts = _pending_prompts()
    prompts.pop(chat_id, None)
    st.session_state.pending_prompts = prompts
    for k in pending_keys:
        pending = st.session_state.get(k)
        if pending and pending.get("chat_id") == chat_id:
            st.session_state[k] = None
    if st.session_state.get("pending_chat_id") == chat_id:
        st.session_state.pending_prompt = None
        st.session_state.pending_chat_id = None


def fail_in_flight(message: str, chat_id=None) -> None:
    """Stop the typing indicator and surface the error on the next paint."""
    st.session_state.last_request_error = message
    clear_in_flight(chat_id)
    st.rerun()


def apply_run_result(chat_id, result, user_text=None):
    _ = user_text
    assistant_text = result.get("assistant_text") or result.get("summary") or "Done."
    add_message(chat_id, "assistant", assistant_text, artifact=result)

    stage = result.get("stage")
    session_id = result.get("session_id") or result.get("run_id")

    deploy_status = {
        "clarifying": "clarifying",
        "plan_review": "plan_pending",
        "spec_approved": "approved",
        "out_of_scope": "out_of_scope",
        "failed": "failed",
    }.get(stage)

    update_chat_run_state(
        chat_id,
        run_id=session_id,
        stage=stage,
        deploy_status=deploy_status,
    )
    st.session_state.focus_chat_input = True


def reconcile_chat_with_api(chat_id) -> None:
    """Align local chat stage with backend session after refresh / resume.

    Skipped in mock mode. On API failure, sets a flag so the UI can show
    a clear banner without blocking the rest of the page.
    """
    if USE_MOCK:
        st.session_state.api_unreachable = False
        return

    session_id = get_active_session_id(chat_id)
    if not session_id:
        return

    # Avoid hammering the API on every widget interaction in the same chat.
    cache_key = f"reconciled_{chat_id}_{session_id}"
    if st.session_state.get(cache_key):
        return

    try:
        state = api_get_session(session_id)
        st.session_state.api_unreachable = False
        st.session_state.pop("api_error_detail", None)
    except OrchestrationApiError as exc:
        st.session_state.api_unreachable = True
        st.session_state.api_error_detail = str(exc)
        return

    st.session_state[cache_key] = True

    if state.get("locked_spec"):
        update_chat_run_state(
            chat_id,
            run_id=session_id,
            stage="spec_approved",
            deploy_status="approved",
        )
    elif state.get("latest_plan"):
        update_chat_run_state(
            chat_id,
            run_id=session_id,
            stage="plan_review",
            deploy_status="plan_pending",
        )


def render_clarifying(artifact):
    questions = artifact.get("clarifying_questions") or []
    if questions:
        for q in questions:
            render_assistant_message(q)
    else:
        text = artifact.get("assistant_text") or ""
        if text:
            render_assistant_message(text)


def render_plan_review(message_id, artifact, chat_id):
    session_id = artifact.get("session_id") or artifact.get("run_id")
    busy = is_chat_busy(chat_id)

    messages = get_messages(chat_id)
    latest_plan_msg_id = None
    for m in reversed(messages):
        art = m.get("artifact") or {}
        if art.get("stage") == "plan_review":
            latest_plan_msg_id = m["id"]
            break

    chat = get_chat(chat_id)
    chat_stage = (chat or {}).get("stage")
    # Only the current open plan is interactive. After approve/reject (or a
    # newer plan), freeze placeholders and hide actions on this card.
    is_active_plan = (
        chat_stage == "plan_review" and message_id == latest_plan_msg_id
    )
    controls_enabled = is_active_plan and not busy

    if artifact.get("summary"):
        st.markdown(artifact["summary"])

    placeholders = artifact.get("placeholders") or []
    # Deduplicate while preserving order — duplicate tokens would collide on key=.
    seen_tokens: list[str] = []
    for token in placeholders:
        if token and token not in seen_tokens:
            seen_tokens.append(token)
    overrides = {}
    if seen_tokens:
        st.markdown("**Placeholders (optional overrides before approve)**")
        for idx, token in enumerate(seen_tokens):
            overrides[token] = st.text_input(
                token,
                value="",
                key=f"ph_{chat_id}_{message_id}_{idx}_{token}",
                placeholder=f"Value for {token}",
                disabled=not controls_enabled,
            )

    # Collapsible diagram only — plan JSON is internal, not shown to users.
    with st.expander("Diagram", expanded=True):
        dot = artifact.get("graph_dot")
        if dot:
            head = (dot or "").lstrip()[:32].lower()
            if head.startswith("flowchart") or head.startswith("statediagram"):
                try:
                    import streamlit_mermaid as stmd
                    stmd.st_mermaid(dot, height=520)
                except Exception:
                    st.markdown(f"```mermaid\n{dot}\n```")
            else:
                try:
                    st.graphviz_chart(dot)
                except Exception:
                    st.code(dot)
        else:
            st.info(
                "No diagram was returned for this plan. "
                "You can still approve or request changes."
            )

    if not is_active_plan:
        st.caption("This plan is no longer active — placeholders and actions are locked.")
        return

    if busy:
        st.caption("Working on your request… actions are paused until it finishes.")

    feedback = st.text_area(
        "Request changes",
        key=f"reject_fb_{message_id}",
        placeholder="Optional: describe changes, then Reject — or Approve as-is.",
        height=80,
        disabled=not controls_enabled,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="approve-btn-marker plan-actions-marker"></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Approve plan",
            key=f"approve_plan_{message_id}",
            type="primary",
            use_container_width=True,
            disabled=not controls_enabled,
        ):
            clean = {k: v.strip() for k, v in overrides.items() if v and v.strip()}
            # Show the user bubble immediately (once). process_pending_approve
            # must NOT add_message again or the action appears twice.
            if clean:
                lines = ["Approve plan", ""] + [
                    f"{k} = {v}" for k, v in clean.items()
                ]
                user_text = "\n".join(lines)
            else:
                user_text = "Approve plan"
            add_message(chat_id, "user", user_text)
            st.session_state.pending_approve = {
                "chat_id": chat_id,
                "session_id": session_id,
                "overrides": clean,
            }
            st.rerun()

    with c2:
        st.markdown(
            '<div class="reject-btn-marker plan-actions-marker"></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Reject / request changes",
            key=f"reject_plan_{message_id}",
            type="secondary",
            use_container_width=True,
            disabled=not controls_enabled,
        ):
            if not (feedback or "").strip():
                st.warning("Enter feedback first.")
            else:
                queue_user_prompt(chat_id, feedback.strip())


def _find_todo_placeholders_in_value(node, seen=None) -> list[str]:
    """Walk any JSON-like value and collect unique TODO_[A-Z0-9_]+ tokens.

    Plan-level remaining_placeholders only cover extraction/plan text.
    Enrichment + ASL generation often introduce additional tokens
    (e.g. TODO_UNKNOWN_INTEGRATION_ARN, TableName TODOs) that only
    exist inside the ASL document — deploy pre-checks scan the ASL,
    so the fill form must too.
    """
    import re

    if seen is None:
        seen = []
    pattern = re.compile(r"\bTODO_[A-Z0-9_]+\b")
    if isinstance(node, str):
        for match in pattern.findall(node):
            if match not in seen:
                seen.append(match)
    elif isinstance(node, dict):
        for v in node.values():
            _find_todo_placeholders_in_value(v, seen)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _find_todo_placeholders_in_value(v, seen)
    return seen


def _latest_spec_approved_message_id(chat_id):
    """Return the message id of the newest spec_approved artifact in this chat."""
    messages = get_messages(chat_id)
    for m in reversed(messages):
        art = m.get("artifact") or {}
        if art.get("stage") == "spec_approved":
            return m["id"]
    return None


def render_enrichment_question(message_id, artifact, chat_id, controls_enabled: bool):
    """UI for Stage 3.5 enrichment clarification (pause during /approve)."""
    question = (artifact.get("enrichment_question") or "").strip()
    if not question:
        return

    session_id = artifact.get("session_id") or artifact.get("run_id")

    st.warning("Enrichment needs a clarification before the workflow spec can be completed.")
    st.markdown(question)

    if not controls_enabled:
        st.caption("This enrichment question is no longer active — answer on the latest approved card.")
        return

    answer = st.text_area(
        "Your answer",
        key=f"enrich_ans_{message_id}",
        placeholder="Describe the structural decision enrichment is asking about…",
        height=100,
        disabled=not controls_enabled,
    )

    st.markdown(
        '<div class="approve-btn-marker plan-actions-marker"></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Submit enrichment answer",
        key=f"enrich_submit_{message_id}",
        type="primary",
        use_container_width=True,
        disabled=not controls_enabled or not session_id,
    ):
        if not (answer or "").strip():
            st.warning("Enter an answer first.")
        else:
            # Show the answer in chat immediately; process_pending_enrichment
            # must not add_message again.
            add_message(chat_id, "user", answer.strip())
            st.session_state.pending_enrichment = {
                "chat_id": chat_id,
                "session_id": session_id,
                "answer": answer.strip(),
            }
            st.rerun()


def render_spec_approved(message_id, artifact, chat_id):
    """Approved plan card: enrichment pause UI, then ASL graph + definition + deploy.

    workflow_spec stays backend-only — not shown here. Validation summary,
    regenerate/revalidate, and Stage 5 deploy controls are interactive on the
    latest approved message only.
    """
    st.success("Plan approved and locked on the server.")

    enrichment_question = artifact.get("enrichment_question")
    enrichment_error = artifact.get("enrichment_error")
    asl = artifact.get("asl") or artifact.get("asl_json")
    asl_error = artifact.get("asl_error")
    session_id = artifact.get("session_id") or artifact.get("run_id")
    busy = is_chat_busy(chat_id)

    is_active = message_id == _latest_spec_approved_message_id(chat_id)
    controls_enabled = is_active and not busy and bool(session_id)

    # Prefer API-reported spec_placeholders (workflow_spec TODOs). Fall back
    # to plan remaining + ASL/validation scans so older messages still work.
    remaining: list[str] = []
    for token in list(artifact.get("spec_placeholders") or []):
        if token and token not in remaining:
            remaining.append(token)
    if not remaining:
        for token in (
            list(artifact.get("remaining_placeholders") or [])
            + list(artifact.get("placeholders") or [])
        ):
            if token and token not in remaining:
                remaining.append(token)
        if asl:
            for token in _find_todo_placeholders_in_value(asl):
                if token not in remaining:
                    remaining.append(token)
        vals = artifact.get("validations") or {}
        for token in vals.get("unresolved_placeholders") or []:
            if token and token not in remaining:
                remaining.append(token)

    # --- Fill remaining placeholders (re-approve with overrides → new ASL) ---
    # Deploy pre-checks reject leftover TODO_ tokens. Users can supply values
    # here after the first approve; we call POST /approve again so enrichment
    # + Stage 4 re-run with a fully resolved plan.
    if remaining and is_active:
        st.warning(
            f"**{len(remaining)} unresolved placeholder(s)** must be filled "
            "before deploy will pass pre-checks."
        )
        with st.expander("Fill placeholders", expanded=True):
            st.caption(
                "Provide concrete values (resource names, ARNs, bucket keys, etc.). "
                "Applying them re-runs enrichment and regenerates ASL."
            )
            ph_overrides: dict[str, str] = {}
            for idx, token in enumerate(remaining):
                ph_overrides[token] = st.text_input(
                    token,
                    value="",
                    key=f"post_ph_{chat_id}_{message_id}_{idx}_{token}",
                    placeholder=f"Value for {token}",
                    disabled=not controls_enabled,
                )
            filled = {
                k: v.strip() for k, v in ph_overrides.items() if v and v.strip()
            }
            missing = [t for t in remaining if t not in filled]
            if missing:
                st.caption(
                    f"{len(missing)} still empty — fill all tokens, then apply."
                )
            apply_label = (
                "Apply placeholders & regenerate ASL"
                if filled
                else "Apply placeholders & regenerate ASL"
            )
            if st.button(
                apply_label,
                key=f"apply_placeholders_{message_id}",
                type="primary",
                use_container_width=True,
                disabled=not controls_enabled or not filled,
            ):
                if missing:
                    st.warning(
                        "Some placeholders are still empty: "
                        + ", ".join(f"`{t}`" for t in missing)
                        + ". You can apply a partial set, but deploy will keep "
                        "blocking until every TODO_ is gone."
                    )
                lines = ["Apply placeholders"] + [
                    f"{k} = {v}" for k, v in filled.items()
                ]
                add_message(chat_id, "user", "\n".join(lines))
                st.session_state.pending_fill_placeholders = {
                    "chat_id": chat_id,
                    "session_id": session_id,
                    "overrides": filled,
                }
                st.rerun()
    elif remaining and not is_active:
        st.warning("Unresolved placeholders (read-only on older cards):")
        for p in remaining:
            st.markdown(f"- `{p}`")

    # --- Enrichment question section (Stage 3.5 pause) ---
    if enrichment_question:
        render_enrichment_question(message_id, artifact, chat_id, controls_enabled)
        if busy:
            st.caption("Working on your answer… actions are paused until it finishes.")
        st.info("ASL will be generated automatically once enrichment completes.")
        return

    if enrichment_error:
        st.error(f"Enrichment failed: {enrichment_error}")
        st.caption(
            "The plan is still locked. Fix enrichment (or re-approve) before ASL "
            "can be generated."
        )
        return

    if asl_error:
        st.error(f"ASL generation error: {asl_error}")

    # --- Validation summary ---
    vals = artifact.get("validations") or {}
    if vals:
        ok = vals.get("ok")
        errs = vals.get("errors") or []
        if ok is True:
            st.success("ASL validations passed.")
        elif ok is False:
            st.warning(f"ASL has {len(errs)} validation issue(s).")
            for e in errs[:8]:
                if isinstance(e, dict):
                    st.markdown(
                        f"- `[{e.get('group')}/{e.get('code')}]` {e.get('message')}"
                    )
                else:
                    st.markdown(f"- {e}")
        aws = vals.get("aws_details") or {}
        if aws.get("skipped"):
            st.caption(f"AWS ValidateStateMachine skipped: {aws.get('reason') or 'n/a'}")
        elif aws.get("attempted") and vals.get("aws_validate_ok") is False:
            st.caption("AWS ValidateStateMachine reported issues (see errors above).")

    # --- ASL graph (console-style) ---
    dot = artifact.get("graph_dot")
    if asl and not dot:
        try:
            from workflow import build_graphviz_dot

            dot = build_graphviz_dot(asl)
        except Exception:
            dot = None

    if dot:
        with st.expander("ASL graph", expanded=True):
            try:
                st.graphviz_chart(dot)
            except Exception as exc:
                st.warning(f"Could not render graph: {exc}")
                st.code(dot, language="dot")
    elif asl:
        st.caption("ASL is present but no graph DOT was returned.")

    # --- ASL JSON (downloadable definition from backend) ---
    if asl:
        with st.expander("ASL definition (JSON)", expanded=False):
            st.json(asl)
            # Download bytes come from GET /session/{id}/asl/download on the
            # API — not from serializing the in-memory UI artifact.
            if not session_id:
                st.caption("No session id — cannot download from backend.")
            elif not is_active:
                st.caption("Only the latest approved plan can download ASL.")
            else:
                from workflow import USE_MOCK, api_download_asl, OrchestrationApiError

                if USE_MOCK:
                    # Mock path has no HTTP API; fall back to local bytes.
                    name = "workflow"
                    if isinstance(asl, dict) and asl.get("Comment"):
                        name = str(asl["Comment"]).split(":")[0].strip() or name
                    st.download_button(
                        "Download ASL JSON",
                        data=json.dumps(asl, indent=2),
                        file_name=f"{name}.asl.json",
                        mime="application/json",
                        key=f"download_asl_{message_id}",
                    )
                else:
                    cache_key = f"asl_download_{session_id}"
                    cached = st.session_state.get(cache_key)
                    if cached is None:
                        try:
                            content, filename = api_download_asl(session_id)
                            st.session_state[cache_key] = (content, filename)
                            cached = (content, filename)
                        except OrchestrationApiError as exc:
                            st.warning(f"Could not fetch ASL from backend: {exc}")
                            cached = None
                    if cached is not None:
                        content, filename = cached
                        st.download_button(
                            "Download ASL JSON",
                            data=content,
                            file_name=filename,
                            mime="application/json",
                            key=f"download_asl_{message_id}",
                        )
    else:
        st.info("ASL not generated yet.")

    # --- ASL review: Approve / Reject (not regenerate/revalidate) ---
    asl_approved_key = f"asl_approved_{session_id}" if session_id else None
    asl_approved = bool(
        (asl_approved_key and st.session_state.get(asl_approved_key))
        or artifact.get("asl_approved")
        or (
            isinstance(artifact.get("deployment"), dict)
            and artifact["deployment"].get("state_machine_arn")
        )
    )
    deploy_prompt_key = f"deploy_prompt_{session_id}" if session_id else None
    show_deploy_form = bool(
        deploy_prompt_key and st.session_state.get(deploy_prompt_key)
    )

    # Cannot review ASL until placeholders are cleared and ASL exists.
    can_review_asl = bool(asl) and not remaining and not enrichment_question

    if is_active and session_id and can_review_asl and not asl_approved:
        st.markdown("#### Review ASL")
        st.caption(
            "Approve if this definition looks right. Reject to request changes "
            "(sends feedback and rebuilds the plan)."
        )
        reject_fb = st.text_area(
            "Reject feedback (optional unless you reject)",
            key=f"asl_reject_fb_{message_id}",
            placeholder="What should change in the ASL / workflow?",
            height=70,
            disabled=not controls_enabled,
        )
        c_ok, c_no = st.columns(2)
        with c_ok:
            st.markdown(
                '<div class="approve-btn-marker plan-actions-marker"></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Approve ASL",
                key=f"approve_asl_{message_id}",
                type="primary",
                use_container_width=True,
                disabled=not controls_enabled,
            ):
                add_message(chat_id, "user", "Approve ASL")
                st.session_state[asl_approved_key] = True
                st.session_state[deploy_prompt_key] = True
                # Record a lightweight assistant turn so the thread shows the decision.
                apply_run_result(
                    chat_id,
                    {
                        "session_id": session_id,
                        "run_id": session_id,
                        "stage": "spec_approved",
                        "assistant_text": (
                            "ASL **approved**. Do you want to deploy this workflow "
                            "to AWS Step Functions?"
                        ),
                        "asl": asl,
                        "asl_json": asl,
                        "graph_dot": artifact.get("graph_dot"),
                        "validations": artifact.get("validations") or {},
                        "spec_placeholders": [],
                        "remaining_placeholders": [],
                        "placeholders": [],
                        "asl_approved": True,
                        "ask_deploy": True,
                    },
                )
                st.rerun()
        with c_no:
            st.markdown(
                '<div class="reject-btn-marker plan-actions-marker"></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Reject ASL",
                key=f"reject_asl_{message_id}",
                type="secondary",
                use_container_width=True,
                disabled=not controls_enabled,
            ):
                fb = (reject_fb or "").strip()
                if not fb:
                    st.warning("Enter feedback describing what to change, then Reject ASL.")
                else:
                    if asl_approved_key:
                        st.session_state.pop(asl_approved_key, None)
                    if deploy_prompt_key:
                        st.session_state.pop(deploy_prompt_key, None)
                    queue_user_prompt(chat_id, fb)

        if busy:
            st.caption("Working… actions are paused until the current request finishes.")

    elif is_active and session_id and remaining and asl:
        st.info(
            "Fill all placeholders above and apply them before you can approve this ASL."
        )

    # --- After ASL approve: deploy prompt ---
    deployment = artifact.get("deployment")
    if deployment and isinstance(deployment, dict) and deployment.get("state_machine_arn"):
        st.success(
            f"Deployed ({deployment.get('action') or 'ok'}): "
            f"`{deployment.get('state_machine_arn')}`"
        )
        st.caption(
            f"Region `{deployment.get('region')}` · role `{deployment.get('role_arn')}`"
        )

    wants_deploy = asl_approved and (
        show_deploy_form
        or artifact.get("ask_deploy")
        or st.session_state.get(deploy_prompt_key)
    )

    if (
        is_active
        and session_id
        and asl
        and asl_approved
        and wants_deploy
        and not (
            isinstance(deployment, dict) and deployment.get("state_machine_arn")
        )
    ):
        st.markdown("#### Deploy this ASL?")
        st.caption(
            "Creates or updates a Step Functions state machine from the locked ASL."
        )
        c_yes, c_skip = st.columns(2)
        with c_yes:
            if st.button(
                "Yes, deploy",
                key=f"deploy_yes_{message_id}",
                type="primary",
                use_container_width=True,
                disabled=not controls_enabled,
            ):
                st.session_state[f"show_deploy_form_{session_id}"] = True
                st.rerun()
        with c_skip:
            if st.button(
                "Not now",
                key=f"deploy_no_{message_id}",
                use_container_width=True,
                disabled=not controls_enabled,
            ):
                if deploy_prompt_key:
                    st.session_state[deploy_prompt_key] = False
                st.session_state.pop(f"show_deploy_form_{session_id}", None)
                add_message(chat_id, "user", "Not now — skip deploy")
                apply_run_result(
                    chat_id,
                    {
                        "session_id": session_id,
                        "run_id": session_id,
                        "stage": "spec_approved",
                        "assistant_text": (
                            "Okay — deploy skipped. You can deploy later from this "
                            "approved ASL card when you're ready."
                        ),
                        "asl": asl,
                        "asl_json": asl,
                        "graph_dot": artifact.get("graph_dot"),
                        "validations": artifact.get("validations") or {},
                        "asl_approved": True,
                        "ask_deploy": False,
                    },
                )
                st.rerun()

        show_form = st.session_state.get(f"show_deploy_form_{session_id}")
        if show_form:
            with st.expander("Deploy to AWS Step Functions", expanded=True):
                st.caption(
                    "Optional overrides — leave blank to use server defaults "
                    "(plan name slug + AWS_STATE_MACHINE_ROLE_ARN)."
                )
                sm_name = st.text_input(
                    "State machine name (optional)",
                    value="",
                    key=f"deploy_name_{message_id}",
                    placeholder="defaults to plan name slug",
                    disabled=not controls_enabled,
                )
                role_arn = st.text_input(
                    "Execution role ARN (optional)",
                    value="",
                    key=f"deploy_role_{message_id}",
                    placeholder="defaults to server AWS_STATE_MACHINE_ROLE_ARN",
                    disabled=not controls_enabled,
                )
                force = st.checkbox(
                    "Force deploy (skip pre-deploy checks)",
                    value=False,
                    key=f"deploy_force_{message_id}",
                    disabled=not controls_enabled,
                )
                if st.button(
                    "Deploy workflow",
                    key=f"deploy_btn_{message_id}",
                    type="primary",
                    use_container_width=True,
                    disabled=not controls_enabled,
                ):
                    add_message(chat_id, "user", "Deploy workflow")
                    st.session_state.pending_deploy = {
                        "chat_id": chat_id,
                        "session_id": session_id,
                        "state_machine_name": (sm_name or "").strip() or None,
                        "role_arn": (role_arn or "").strip() or None,
                        "force": bool(force),
                    }
                    st.rerun()

    # Show last pre-deploy failure reasons if present on this artifact
    pre_reasons = artifact.get("pre_deploy_reasons")
    if pre_reasons:
        st.error("Pre-deploy checks failed:")
        for r in pre_reasons:
            st.markdown(f"- {r}")


def render_out_of_scope(artifact):
    text = artifact.get("assistant_text") or (
        "This assistant is only designed to help you build AWS Step Functions "
        "workflows. Please describe a workflow to orchestrate."
    )
    st.info(text)


def render_artifact(message_id, artifact, chat_id):
    if not artifact:
        return
    stage = artifact.get("stage") or "unknown"

    if stage == "clarifying":
        render_clarifying(artifact)
    elif stage == "plan_review":
        render_plan_review(message_id, artifact, chat_id)
    elif stage == "spec_approved":
        render_spec_approved(message_id, artifact, chat_id)
    elif stage == "out_of_scope":
        render_out_of_scope(artifact)
    elif stage == "failed":
        st.error(artifact.get("assistant_text") or "Something failed.")
    else:
        st.json(artifact)


def render_chat_thread(chat_id, *, simplify: bool = False) -> None:
    """Paint only this chat's messages.

    ``simplify=True`` (used while an API call is in flight) renders plain
    text bubbles only — no plan diagrams / ASL graphs — so heavy widgets
    from another chat cannot linger in the element tree.
    """
    for message in get_messages(chat_id):
        mid = message.get("id")
        # Unique marker so Streamlit cannot reconcile this block with a
        # message from a different chat at the same list index.
        st.markdown(
            f'<div data-msg="{chat_id}-{mid}" style="display:none"></div>',
            unsafe_allow_html=True,
        )
        if message["role"] == "user":
            render_user_message(message["content"])
            continue

        artifact = message.get("artifact")
        stage = (artifact or {}).get("stage") if artifact else None

        if simplify:
            # Plain text only while waiting — avoids graphviz/expanders.
            text = message.get("content") or ""
            if stage == "clarifying" and artifact:
                render_clarifying(artifact)
            elif text:
                render_assistant_message(text)
            continue

        if stage == "clarifying":
            render_artifact(mid, artifact, chat_id)
            continue
        if stage == "out_of_scope":
            render_assistant_message(message["content"])
            continue
        if stage not in ("plan_review", "spec_approved", "failed"):
            render_assistant_message(message["content"])
            if artifact:
                st.markdown(
                    '<div class="assistant-response">', unsafe_allow_html=True
                )
                render_artifact(mid, artifact, chat_id)
                st.markdown("</div>", unsafe_allow_html=True)
            continue

        st.markdown('<div class="assistant-response">', unsafe_allow_html=True)
        render_assistant_block_start()
        if artifact:
            render_artifact(mid, artifact, chat_id)
        else:
            st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)


def queue_user_prompt(active_chat_id, user_prompt):
    # Only block if THIS chat already has a request in flight. Other chats
    # remain fully independent sessions.
    if is_chat_busy(active_chat_id):
        st.warning("Please wait for the current request to finish…")
        return

    if get_chat(active_chat_id) is None:
        st.session_state.pop("active_chat_id", None)
        ensure_active_chat()
        active_chat_id = st.session_state.active_chat_id

    chat = get_chat(active_chat_id)
    if chat and chat.get("title") in ("New workflow chat", "Untitled workflow"):
        update_chat_title(active_chat_id, create_chat_title(user_prompt))

    add_message(active_chat_id, "user", user_prompt)
    prompts = _pending_prompts()
    prompts[active_chat_id] = user_prompt
    st.session_state.pending_prompts = prompts
    # Hide empty-state example chips immediately for this chat.
    st.session_state[f"examples_dismissed_{active_chat_id}"] = True
    st.session_state.focus_chat_input = True
    st.rerun()


def _clear_reconcile_cache(chat_id) -> None:
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(f"reconciled_{chat_id}_"):
            st.session_state.pop(key, None)


def _finish_in_flight(chat_id) -> None:
    """Persist cleared busy state. Rerun only if the user is still on this chat
    so a mid-request sidebar switch does not paint the wrong thread."""
    clear_in_flight(chat_id)
    _clear_reconcile_cache(chat_id)
    if st.session_state.get("active_chat_id") == chat_id:
        st.rerun()


def process_pending_approve(region, env):
    """Run a queued approve exactly once (guards double-submit).

    The user bubble was already written when the button was clicked —
    do not add_message again here.
    """
    _ = region, env
    pending = st.session_state.get("pending_approve")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    overrides = pending.get("overrides") or {}

    status, result, err = run_with_live_status(
        chat_id, lambda: approve_plan(session_id, overrides=overrides)
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(f"Could not approve the plan. {err}", chat_id=chat_id)
        return
    # ASL may have changed — drop any cached download bytes for this session.
    st.session_state.pop(f"asl_download_{session_id}", None)
    apply_run_result(chat_id, result)
    _finish_in_flight(chat_id)


def process_pending_enrichment(region, env):
    """Run a queued enrichment answer exactly once (guards double-submit).

    The user bubble was already written when Submit was clicked —
    do not add_message again here.
    """
    _ = region, env
    pending = st.session_state.get("pending_enrichment")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    answer = pending.get("answer") or ""

    status, result, err = run_with_live_status(
        chat_id, lambda: answer_enrichment(session_id, answer)
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(f"Could not answer enrichment. {err}", chat_id=chat_id)
        return
    st.session_state.pop(f"asl_download_{session_id}", None)
    apply_run_result(chat_id, result)
    _finish_in_flight(chat_id)



def process_pending_fill_placeholders(region, env):
    """POST /session/{id}/spec/placeholders — resolve workflow_spec TODOs."""
    _ = region, env
    pending = st.session_state.get("pending_fill_placeholders")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    overrides = pending.get("overrides") or {}

    status, result, err = run_with_live_status(
        chat_id, lambda: fill_spec_placeholders(session_id, overrides=overrides)
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(f"Could not apply placeholders. {err}", chat_id=chat_id)
        st.session_state.pop("pending_fill_placeholders", None)
        return

    st.session_state.pop("pending_fill_placeholders", None)
    st.session_state.pop(f"asl_download_{session_id}", None)
    apply_run_result(chat_id, result)
    _finish_in_flight(chat_id)


def process_pending_generate_asl(region, env):
    """Re-run Stage 4 ASL generation for the locked workflow_spec."""
    _ = region, env
    pending = st.session_state.get("pending_generate_asl")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]

    status, result, err = run_with_live_status(
        chat_id, lambda: generate_asl(session_id)
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(f"Could not regenerate ASL. {err}", chat_id=chat_id)
        st.session_state.pop("pending_generate_asl", None)
        return

    st.session_state.pop("pending_generate_asl", None)
    st.session_state.pop(f"asl_download_{session_id}", None)

    # Merge generate payload into a full UI artifact so the thread can re-render.
    art = {
        "session_id": session_id,
        "run_id": session_id,
        "stage": "spec_approved",
        "assistant_text": "ASL regenerated from the locked workflow specification.",
        "asl": result.get("asl"),
        "asl_json": result.get("asl"),
        "graph_dot": result.get("graph_dot"),
        "validations": result.get("validations") or {},
        "remaining_placeholders": [],
        "enrichment_question": None,
        "enrichment_error": None,
        "asl_error": None,
    }
    apply_run_result(chat_id, art)
    _finish_in_flight(chat_id)


def process_pending_revalidate_asl(region, env):
    """Re-run validation groups on the current ASL."""
    _ = region, env
    pending = st.session_state.get("pending_revalidate_asl")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    asl = pending.get("asl")

    status, result, err = run_with_live_status(
        chat_id, lambda: revalidate_asl(asl, session_id=session_id)
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(f"Could not re-validate ASL. {err}", chat_id=chat_id)
        st.session_state.pop("pending_revalidate_asl", None)
        return

    st.session_state.pop("pending_revalidate_asl", None)

    vals = result.get("validations") or {}
    ok = vals.get("ok")
    n_err = len(vals.get("errors") or [])
    summary = (
        "ASL re-validation **passed**."
        if ok
        else f"ASL re-validation finished with **{n_err}** issue(s)."
    )
    art = {
        "session_id": session_id,
        "run_id": session_id,
        "stage": "spec_approved",
        "assistant_text": summary,
        "asl": result.get("asl") or asl,
        "asl_json": result.get("asl") or asl,
        "graph_dot": result.get("graph_dot"),
        "validations": vals,
        "remaining_placeholders": [],
        "enrichment_question": None,
        "enrichment_error": None,
        "asl_error": None,
    }
    apply_run_result(chat_id, art)
    _finish_in_flight(chat_id)


def process_pending_deploy(region, env):
    """Stage 5: create-or-update the state machine from locked ASL."""
    _ = region, env
    pending = st.session_state.get("pending_deploy")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]

    def _do_deploy():
        return deploy_workflow(
            session_id,
            state_machine_name=pending.get("state_machine_name"),
            role_arn=pending.get("role_arn"),
            force=bool(pending.get("force")),
        )

    status, result, err = run_with_live_status(chat_id, _do_deploy)
    if status == "running":
        return
    if status == "error":
        # Surface structured pre-deploy checklist when available.
        if isinstance(err, PreDeployCheckFailed) or (
            hasattr(err, "reasons") and getattr(err, "reasons", None)
        ):
            reasons = list(getattr(err, "reasons", []) or [])
            art = {
                "session_id": session_id,
                "run_id": session_id,
                "stage": "spec_approved",
                "assistant_text": "Pre-deploy checks failed — fix the items below or force deploy.",
                "pre_deploy_reasons": reasons,
                "asl": None,
            }
            # Keep previous ASL on the thread by not wiping; just add the error note.
            apply_run_result(chat_id, art)
            update_chat_run_state(
                chat_id, run_id=session_id, stage="spec_approved", deploy_status="blocked"
            )
        else:
            fail_in_flight(f"Deploy failed. {err}", chat_id=chat_id)
        st.session_state.pop("pending_deploy", None)
        _finish_in_flight(chat_id)
        return

    st.session_state.pop("pending_deploy", None)
    arn = (result or {}).get("state_machine_arn")
    action = (result or {}).get("action") or "ok"
    art = {
        "session_id": session_id,
        "run_id": session_id,
        "stage": "spec_approved",
        "assistant_text": (
            f"Workflow **{action}** on AWS Step Functions.\n\n"
            f"`{arn}`"
        ),
        "deployment": result,
        "remaining_placeholders": [],
        "enrichment_question": None,
        "enrichment_error": None,
        "asl_error": None,
    }
    apply_run_result(chat_id, art)
    update_chat_run_state(
        chat_id, run_id=session_id, stage="spec_approved", deploy_status="deployed"
    )
    _finish_in_flight(chat_id)



def process_user_prompt(active_chat_id, user_prompt, region, env):
    if get_chat(active_chat_id) is None:
        st.session_state.pop("active_chat_id", None)
        ensure_active_chat()
        active_chat_id = st.session_state.active_chat_id

    chat = get_chat(active_chat_id)
    session_id = get_active_session_id(active_chat_id)

    # Brand-new chat: no session exists yet. Create it up front (a fast
    # POST /session call, not the slow LLM work) instead of letting
    # send_message() create it lazily on the background thread — that left
    # nothing for the live-status checklist to poll GET /status with for
    # the entire first message, so it always sat stuck on step one.
    if not session_id and not USE_MOCK:
        try:
            session_id = create_session()
        except OrchestrationApiError as exc:
            fail_in_flight(
                f"Could not reach the orchestration API. Check that the API "
                f"is running and try again. ({exc})",
                chat_id=active_chat_id,
            )
            return
        _remember_session_id(active_chat_id, session_id)

    if chat and chat.get("stage") == "plan_review" and session_id and not USE_MOCK:
        status, result, err = run_with_live_status(
            active_chat_id,
            lambda: reject_plan(session_id, user_prompt, region=region, env=env),
        )
        if status == "running":
            return
        if status == "error":
            fail_in_flight(
                f"Could not send the change request. {err}",
                chat_id=active_chat_id,
            )
            return
        apply_run_result(active_chat_id, result)
        _finish_in_flight(active_chat_id)
        return

    status, result, err = run_with_live_status(
        active_chat_id,
        lambda: send_message(user_prompt, run_id=session_id, region=region, env=env),
    )
    if status == "running":
        return
    if status == "error":
        fail_in_flight(
            "Could not reach the orchestration API. "
            "Check that the API is running and try again. "
            f"({err})",
            chat_id=active_chat_id,
        )
        return
    apply_run_result(active_chat_id, result)
    _finish_in_flight(active_chat_id)


def _chat_input_resize_script(nonce: str, *, focus: bool) -> str:
    """After send, clear leftover expanded height on an empty chat textarea.

    Does not force a tiny fixed height (that made the box look distorted and
    scrollable). Only removes inline height Streamlit left behind, and
    optionally focuses the input. No MutationObserver.
    """
    focus_js = "true" if focus else "false"
    return f"""
        <script>
        (function() {{
          const shouldFocus = {focus_js};
          const doc = window.parent.document;
          let attempts = 0;

          function findInput() {{
            const root = doc.querySelector('[data-testid="stChatInput"]');
            if (!root) return null;
            return root.querySelector("textarea") || root.querySelector("input");
          }}

          function resetIfEmpty(input) {{
            if (!input) return;
            const empty = !input.value || !String(input.value).trim();
            if (!empty) return;
            try {{
              // Remove Streamlit auto-grow inline height; let CSS/layout size it.
              input.style.removeProperty("height");
              input.style.removeProperty("min-height");
              input.style.removeProperty("max-height");
              input.rows = 1;
              // Clear expanded height on immediate wrappers only.
              let el = input.parentElement;
              for (let i = 0; i < 3 && el; i++) {{
                if (el.style) {{
                  el.style.removeProperty("height");
                  el.style.removeProperty("min-height");
                }}
                el = el.parentElement;
              }}
            }} catch (e) {{}}
          }}

          function run() {{
            const input = findInput();
            if (!input) {{
              attempts += 1;
              if (attempts < 40) setTimeout(run, 50);
              return;
            }}
            resetIfEmpty(input);
            if (shouldFocus) {{
              try {{ input.focus({{ preventScroll: true }}); }}
              catch (e) {{ try {{ input.focus(); }} catch (e2) {{}} }}
            }}
          }}

          setTimeout(run, 0);
          setTimeout(run, 150);
          setTimeout(run, 400);
          setTimeout(run, 800);
        }})();
        </script>
        """


def focus_chat_input(active_chat_id):
    """Focus the chat box when requested; collapse expanded empty height after send."""
    want_focus = bool(st.session_state.pop("focus_chat_input", False))
    st.session_state._focus_tick = st.session_state.get("_focus_tick", 0) + 1
    nonce = f"{active_chat_id}-{st.session_state.get('_focus_tick', 0)}-{int(want_focus)}"

    components.html(
        _chat_input_resize_script(nonce, focus=want_focus),
        height=0,
    )


# ----- body -----
# First run of this browser session: restore the chat from ?chat= in the URL
# when present (so a refresh stays on the same conversation). Otherwise open
# a draft. Later reruns keep the active chat via ensure_active_chat.
if "session_bootstrapped" not in st.session_state:
    st.session_state.session_bootstrapped = True
    st.session_state.pop("active_chat_id", None)
    ensure_active_chat()
    st.session_state.focus_chat_input = True
else:
    ensure_active_chat()

if "pending_prompts" not in st.session_state:
    st.session_state.pending_prompts = {}
# Migrate legacy single-pending keys if a hot-reload left them set.
if st.session_state.get("pending_prompt") and st.session_state.get("pending_chat_id"):
    legacy_cid = st.session_state.pending_chat_id
    legacy_prompt = st.session_state.pending_prompt
    prompts = _pending_prompts()
    if legacy_cid not in prompts:
        prompts[legacy_cid] = legacy_prompt
        st.session_state.pending_prompts = prompts
    st.session_state.pending_prompt = None
    st.session_state.pending_chat_id = None
if "pending_approve" not in st.session_state:
    st.session_state.pending_approve = None
if "pending_enrichment" not in st.session_state:
    st.session_state.pending_enrichment = None
if "api_unreachable" not in st.session_state:
    st.session_state.api_unreachable = False

with st.sidebar:
    st.title("Workflow Assistant")

    if USE_MOCK:
        st.caption("Mock mode — set USE_MOCK_API=false for real API")
    else:
        st.caption("Live API mode")

    st.selectbox("Environment", options=["dev", "test"], key="deploy_env")
    st.selectbox(
        "AWS region",
        options=["ap-south-1", "us-east-1", "eu-west-1"],
        key="aws_region",
    )

    if st.button("New Chat", type="primary", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="sidebar-heading">CHAT HISTORY</div>',
        unsafe_allow_html=True,
    )

    with st.container(height=400, border=False):
        st.markdown(
            '<div class="sidebar-history-marker" style="display:none"></div>',
            unsafe_allow_html=True,
        )

        for chat in get_chats():
            if is_draft_chat(chat["id"]):
                continue
            render_sidebar_chat(chat)
        st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)

active_chat_id = st.session_state.active_chat_id
if not get_chat(active_chat_id):
    st.session_state.pop("active_chat_id", None)
    ensure_active_chat()
    active_chat_id = st.session_state.active_chat_id

region = st.session_state.get("aws_region", "ap-south-1")
env = st.session_state.get("deploy_env", "dev")

st.title("Workflow Assistant")
st.caption("Chat → plan → approve plan → fill placeholders → approve ASL → deploy")

if st.session_state.get("api_unreachable") and not USE_MOCK:
    detail = st.session_state.get("api_error_detail") or "API unreachable"
    st.warning(
        "Orchestration API is not reachable. Chat history still loads from local "
        f"storage; approve / send will fail until the API is back. ({detail})"
    )

# Single replaceable slot for the whole thread. st.empty() discards the
# previous run's elements so switching chats cannot leave another chat's
# last bubble on screen.
_thread_slot = st.empty()
_typing_slot = st.empty()
_examples_slot = st.empty()

_busy_now = is_chat_busy(active_chat_id)

if _busy_now:
    # In-flight: plain bubbles only + typing, then run the API. Never paint
    # the full interactive thread in the same run (avoids duplicate widget keys).
    with _thread_slot.container():
        render_chat_thread(active_chat_id, simplify=True)

    # IMPORTANT: consume any finished background job BEFORE rendering the
    # live-status fragment. render_live_status() calls st.rerun() as soon as
    # it sees a job whose status is no longer "running" — and st.rerun()
    # halts this script immediately. If the fragment ran first, a finished
    # job would never reach process_pending_* (which is what actually pops
    # the job, applies the result, and adds the assistant message), so the
    # app would loop forever re-triggering reruns on a job nobody consumes:
    # a spinner that never resolves and a chat message that never appears.
    pending_approve = st.session_state.get("pending_approve")
    if pending_approve and pending_approve.get("chat_id") == active_chat_id:
        process_pending_approve(region=region, env=env)

    pending_enrichment = st.session_state.get("pending_enrichment")
    if pending_enrichment and pending_enrichment.get("chat_id") == active_chat_id:
        process_pending_enrichment(region=region, env=env)

    pending_fill = st.session_state.get("pending_fill_placeholders")
    if pending_fill and pending_fill.get("chat_id") == active_chat_id:
        process_pending_fill_placeholders(region=region, env=env)

    pending_generate = st.session_state.get("pending_generate_asl")
    if pending_generate and pending_generate.get("chat_id") == active_chat_id:
        process_pending_generate_asl(region=region, env=env)

    pending_revalidate = st.session_state.get("pending_revalidate_asl")
    if pending_revalidate and pending_revalidate.get("chat_id") == active_chat_id:
        process_pending_revalidate_asl(region=region, env=env)

    pending_deploy = st.session_state.get("pending_deploy")
    if pending_deploy and pending_deploy.get("chat_id") == active_chat_id:
        process_pending_deploy(region=region, env=env)

    pending_prompts = _pending_prompts()
    if active_chat_id in pending_prompts:
        process_user_prompt(
            active_chat_id,
            pending_prompts[active_chat_id],
            region=region,
            env=env,
        )

    # Processors call st.rerun() as soon as a job completes. If we're still
    # here, the job (if any) is genuinely still running, so it's now safe to
    # show the polling indicator without racing the job's completion.
    with _typing_slot.container():
        render_live_status(
            active_chat_id,
            get_active_session_id(active_chat_id),
            fallback_label=typing_label_for_chat(active_chat_id),
        )

    # Processors call st.rerun() on success. If we are still here, either the
    # request is still marked busy or the user switched away — stop so we do
    # not double-render the interactive thread in this same run.
    try:
        st.chat_input("Please wait…", disabled=True)
    except TypeError:
        st.chat_input("Please wait…")
    st.stop()

# Idle path — full thread once, examples, input.
_typing_slot.empty()

if not is_chat_busy(active_chat_id):
    reconcile_chat_with_api(active_chat_id)

with _thread_slot.container():
    render_chat_thread(active_chat_id, simplify=False)

messages = get_messages(active_chat_id)
has_user = any(m["role"] == "user" for m in messages)
_examples_dismissed = st.session_state.get(f"examples_dismissed_{active_chat_id}", False)

if not has_user and not _examples_dismissed:
    with _examples_slot.container():
        st.markdown("##### Try an example")
        for i, example in enumerate(EXAMPLE_PROMPTS):
            if st.button(
                example,
                key=f"ex_{active_chat_id}_{i}",
                use_container_width=True,
            ):
                queue_user_prompt(active_chat_id, example)
else:
    _examples_slot.empty()

last_error = st.session_state.pop("last_request_error", None)
if last_error:
    st.error(last_error)

busy = is_chat_busy(active_chat_id)
chat_placeholder = (
    "Please wait…" if busy else "Describe a workflow, or answer the question…"
)
try:
    user_prompt = st.chat_input(chat_placeholder, disabled=busy)
except TypeError:
    user_prompt = st.chat_input(chat_placeholder)
focus_chat_input(active_chat_id)

if user_prompt and not busy:
    queue_user_prompt(active_chat_id, user_prompt)