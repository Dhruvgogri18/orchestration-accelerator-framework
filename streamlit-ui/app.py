import json
import re

import streamlit as st
import streamlit.components.v1 as components

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
    answer_enrichment,
    api_get_session,
    approve_plan,
    reject_plan,
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
            "ASL generation and deploy are not available until those APIs exist."
        ),
    )


def set_active_chat(chat_id, focus: bool = True) -> None:
    """Switch the open chat and persist it in the URL so a browser refresh
    stays on the same conversation instead of opening a new draft."""
    st.session_state.active_chat_id = chat_id
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
    chat = get_chat(chat_id)
    if chat and chat.get("run_id"):
        return chat["run_id"]
    for msg in reversed(get_messages(chat_id)):
        art = msg.get("artifact") or {}
        sid = art.get("session_id") or art.get("run_id")
        if sid:
            return sid
    return None


def create_new_chat():
    """Always open an independent draft chat (new session). Reuse an existing
    unused draft only when it is already the active one — never steal focus
    away from a chat that is still waiting on a response."""
    active_id = st.session_state.get("active_chat_id")
    if active_id and get_chat(active_id) is not None and is_draft_chat(active_id):
        set_active_chat(active_id)
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
# Desktop computer: monitor + stand + keyboard.
_ASSISTANT_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="3" width="18" height="12" rx="1.5"/>'
    '<path d="M8 21h8"/>'
    '<path d="M12 15v6"/>'
    '<path d="M7 18h10"/>'
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


def typing_label_for_chat(chat_id) -> str:
    """Short, context-aware label while the backend works."""
    if st.session_state.get("pending_approve"):
        return "Approving…"
    if st.session_state.get("pending_enrichment"):
        return "Enriching…"
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
    """True while a message, approve, enrichment, or reject is in flight for this chat only."""
    if chat_id is None:
        return False
    if chat_id in _pending_prompts():
        return True
    pending_approve = st.session_state.get("pending_approve")
    if pending_approve and pending_approve.get("chat_id") == chat_id:
        return True
    pending_enrichment = st.session_state.get("pending_enrichment")
    if pending_enrichment and pending_enrichment.get("chat_id") == chat_id:
        return True
    return False


def is_ui_busy() -> bool:
    """True while any chat has a request in flight (legacy / global checks)."""
    if _pending_prompts():
        return True
    if st.session_state.get("pending_approve"):
        return True
    return bool(st.session_state.get("pending_enrichment"))


def clear_in_flight(chat_id=None):
    """Clear in-flight state. If chat_id is given, only clear that chat."""
    if chat_id is None:
        st.session_state.pending_prompts = {}
        st.session_state.pending_approve = None
        st.session_state.pending_enrichment = None
        # Legacy keys (older runs / partial upgrades)
        st.session_state.pending_prompt = None
        st.session_state.pending_chat_id = None
        return

    prompts = _pending_prompts()
    prompts.pop(chat_id, None)
    st.session_state.pending_prompts = prompts
    pending_approve = st.session_state.get("pending_approve")
    if pending_approve and pending_approve.get("chat_id") == chat_id:
        st.session_state.pending_approve = None
    pending_enrichment = st.session_state.get("pending_enrichment")
    if pending_enrichment and pending_enrichment.get("chat_id") == chat_id:
        st.session_state.pending_enrichment = None
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
    overrides = {}
    if placeholders:
        st.markdown("**Placeholders (optional overrides before approve)**")
        for token in placeholders:
            overrides[token] = st.text_input(
                token,
                value="",
                key=f"ph_{message_id}_{token}",
                placeholder=f"Value for {token}",
                disabled=not controls_enabled,
            )

    # Diagram in a tab (JSON reserved for ASL later). Actions sit below.
    tab_diagram = st.tabs(["Diagram"])[0]
    with tab_diagram:
        dot = artifact.get("graph_dot")
        if dot:
            st.graphviz_chart(dot)
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
            # Queue approve so a double-click cannot fire twice.
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
            st.session_state.pending_enrichment = {
                "chat_id": chat_id,
                "session_id": session_id,
                "answer": answer.strip(),
            }
            st.rerun()


def render_spec_approved(message_id, artifact, chat_id):
    st.success("Plan approved and locked on the server.")
    remaining = artifact.get("remaining_placeholders") or artifact.get("placeholders") or []
    if remaining:
        st.warning("Unresolved placeholders:")
        for p in remaining:
            st.markdown(f"- `{p}`")

    enrichment_question = artifact.get("enrichment_question")
    enrichment_error = artifact.get("enrichment_error")
    workflow_spec = artifact.get("workflow_spec")
    session_id = artifact.get("session_id") or artifact.get("run_id")
    busy = is_chat_busy(chat_id)

    # Only the latest approved card is interactive for enrichment answers.
    is_active = message_id == _latest_spec_approved_message_id(chat_id)
    controls_enabled = is_active and not busy and bool(session_id)

    # --- Enrichment question section (Stage 3.5 pause) ---
    if enrichment_question:
        render_enrichment_question(message_id, artifact, chat_id, controls_enabled)
        if busy:
            st.caption("Working on your answer… actions are paused until it finishes.")
        st.info("ASL generation and deploy will appear here when those APIs are available.")
        return

    if enrichment_error:
        st.error(f"Enrichment failed: {enrichment_error}")
        st.caption(
            "The plan is still locked. You can continue from this approved plan "
            "once enrichment is fixed; ASL generation is not available yet."
        )
    elif workflow_spec:
        tabs = st.tabs(["Workflow spec", "Resolved plan"])
        with tabs[0]:
            st.json(workflow_spec)
        with tabs[1]:
            plan = artifact.get("plan")
            if plan:
                st.json(plan)
            else:
                st.info("No separate resolved plan payload.")
    else:
        plan = artifact.get("plan")
        if plan:
            with st.expander("Resolved plan", expanded=False):
                st.json(plan)

    # ASL generation / deploy reserved for later API stages.
    st.info("ASL generation and deploy will appear here when those APIs are available.")


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


def process_pending_approve(region, env):
    """Run a queued approve exactly once (guards double-submit)."""
    _ = region, env
    pending = st.session_state.get("pending_approve")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    overrides = pending.get("overrides") or {}

    if overrides:
        lines = ["Approve plan", ""] + [f"{k} = {v}" for k, v in overrides.items()]
        user_text = "\n".join(lines)
    else:
        user_text = "Approve plan"
    add_message(chat_id, "user", user_text)

    try:
        # Spinner forces Streamlit to flush the typing row above; chrome is CSS-hidden.
        with st.spinner("Approving…"):
            result = approve_plan(session_id, overrides=overrides)
        apply_run_result(chat_id, result)
        # Force a fresh reconcile next time this session is viewed.
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(f"reconciled_{chat_id}_"):
                st.session_state.pop(key, None)
        clear_in_flight(chat_id)
        st.rerun()
    except OrchestrationApiError as exc:
        fail_in_flight(f"Could not approve the plan. {exc}", chat_id=chat_id)


def process_pending_enrichment(region, env):
    """Run a queued enrichment answer exactly once (guards double-submit)."""
    _ = region, env
    pending = st.session_state.get("pending_enrichment")
    if not pending:
        return

    chat_id = pending["chat_id"]
    session_id = pending["session_id"]
    answer = pending.get("answer") or ""

    add_message(chat_id, "user", answer)

    try:
        with st.spinner("Enriching…"):
            result = answer_enrichment(session_id, answer)
        apply_run_result(chat_id, result)
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(f"reconciled_{chat_id}_"):
                st.session_state.pop(key, None)
        clear_in_flight(chat_id)
        st.rerun()
    except OrchestrationApiError as exc:
        fail_in_flight(f"Could not answer enrichment. {exc}", chat_id=chat_id)


def process_user_prompt(active_chat_id, user_prompt, region, env):
    if get_chat(active_chat_id) is None:
        st.session_state.pop("active_chat_id", None)
        ensure_active_chat()
        active_chat_id = st.session_state.active_chat_id

    chat = get_chat(active_chat_id)
    session_id = get_active_session_id(active_chat_id)

    if chat and chat.get("stage") == "plan_review" and session_id and not USE_MOCK:
        try:
            with st.spinner("Updating the plan…"):
                result = reject_plan(session_id, user_prompt, region=region, env=env)
            apply_run_result(active_chat_id, result)
            for key in list(st.session_state.keys()):
                if isinstance(key, str) and key.startswith(
                    f"reconciled_{active_chat_id}_"
                ):
                    st.session_state.pop(key, None)
            clear_in_flight(active_chat_id)
            st.rerun()
            return
        except OrchestrationApiError as exc:
            fail_in_flight(
                f"Could not send the change request. {exc}",
                chat_id=active_chat_id,
            )
            return

    try:
        with st.spinner("Thinking…"):
            result = send_message(
                user_prompt,
                run_id=session_id,
                region=region,
                env=env,
            )
        apply_run_result(active_chat_id, result)
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(f"reconciled_{active_chat_id}_"):
                st.session_state.pop(key, None)
        clear_in_flight(active_chat_id)
        st.rerun()
    except OrchestrationApiError as exc:
        fail_in_flight(
            "Could not reach the orchestration API. "
            "Check that the API is running and try again. "
            f"({exc})",
            chat_id=active_chat_id,
        )


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

# Align local stage with backend session (refresh / resume safety).
# Skip only while THIS chat is mid-request so we don't race the in-flight call.
if not is_chat_busy(active_chat_id):
    reconcile_chat_with_api(active_chat_id)

st.title("Workflow Assistant")
st.caption("Chat → clarify → plan review → approve / reject")

if st.session_state.get("api_unreachable") and not USE_MOCK:
    detail = st.session_state.get("api_error_detail") or "API unreachable"
    st.warning(
        "Orchestration API is not reachable. Chat history still loads from local "
        f"storage; approve / send will fail until the API is back. ({detail})"
    )

messages = get_messages(active_chat_id)
# Any user message (typed or from clicking an example) means the empty-state
# example chips should not be shown again in this chat.
has_user = any(m["role"] == "user" for m in messages)
busy_early = is_chat_busy(active_chat_id)

for message in messages:
    if message["role"] == "user":
        render_user_message(message["content"])
    else:
        artifact = message.get("artifact")
        stage = (artifact or {}).get("stage") if artifact else None

        # Clarifying / out-of-scope: single left-side chat-style reply.
        if stage == "clarifying":
            render_artifact(message["id"], artifact, active_chat_id)
            continue
        if stage == "out_of_scope":
            render_assistant_message(message["content"])
            continue

        # Plain assistant text (welcome, generic) — chat bubble.
        if stage not in ("plan_review", "spec_approved", "failed"):
            render_assistant_message(message["content"])
            if artifact:
                st.markdown('<div class="assistant-response">', unsafe_allow_html=True)
                render_artifact(message["id"], artifact, active_chat_id)
                st.markdown("</div>", unsafe_allow_html=True)
            continue

        # Richer stages: labeled assistant block (plan / approve / error).
        st.markdown('<div class="assistant-response">', unsafe_allow_html=True)
        render_assistant_block_start()
        if artifact:
            render_artifact(message["id"], artifact, active_chat_id)
        else:
            st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

# Example chips: after messages, BEFORE typing + process_user_prompt.
# Must run before the spinner so Streamlit clears prior-run buttons and they
# never sit under "Thinking…". Hide once the user has sent anything.
_examples_box = st.empty()
if not has_user and not busy_early:
    with _examples_box.container():
        st.markdown("##### Try an example")
        for i, example in enumerate(EXAMPLE_PROMPTS):
            if st.button(
                example,
                key=f"ex_{active_chat_id}_{i}",
                use_container_width=True,
            ):
                queue_user_prompt(active_chat_id, example)
else:
    _examples_box.empty()

region = st.session_state.get("aws_region", "ap-south-1")
env = st.session_state.get("deploy_env", "dev")

# Show API/request errors without leaving the typing indicator stuck on screen.
last_error = st.session_state.pop("last_request_error", None)
if last_error:
    st.error(last_error)

# Chat-native typing row (above the hidden spinner that flushes the UI).
# Only for the chat that actually has work in flight.
if is_chat_busy(active_chat_id):
    render_typing_indicator(typing_label_for_chat(active_chat_id))

# Process approve for the open chat (or any queued approve tied to it).
pending_approve = st.session_state.get("pending_approve")
if pending_approve and pending_approve.get("chat_id") == active_chat_id:
    process_pending_approve(region=region, env=env)

# Process enrichment answer for the open chat.
pending_enrichment = st.session_state.get("pending_enrichment")
if pending_enrichment and pending_enrichment.get("chat_id") == active_chat_id:
    process_pending_enrichment(region=region, env=env)

# Process this chat's queued prompt only — other chats keep their own queue
# and run when the user switches back to them.
pending_prompts = _pending_prompts()
if active_chat_id in pending_prompts:
    process_user_prompt(
        active_chat_id,
        pending_prompts[active_chat_id],
        region=region,
        env=env,
    )

busy = is_chat_busy(active_chat_id)

chat_placeholder = (
    "Please wait…" if busy else "Describe a workflow, or answer the question…"
)
# disabled= may not exist on older Streamlit; fall back gracefully.
try:
    user_prompt = st.chat_input(chat_placeholder, disabled=busy)
except TypeError:
    user_prompt = st.chat_input(chat_placeholder)
focus_chat_input(active_chat_id)

if user_prompt and not busy:
    queue_user_prompt(active_chat_id, user_prompt)
