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
/* Main chat area */
.block-container {
  max-width: 920px;
  padding-top: 2rem;
  padding-bottom: 8rem;
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
  margin: 0.55rem 0 !important;
}
.sidebar-heading {
  margin: 0.1rem 0 0.3rem !important;
  font-size: 0.68rem !important;
  color: #6b7280;
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
  background: white;
  border: 1px solid #d8dee9;
  border-radius: 18px;
  box-shadow: 0 8px 25px rgba(15, 23, 42, 0.16);
}
[data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stChatInput"] {
  left: 50%;
  width: min(820px, calc(100vw - 3rem));
}
[data-testid="stChatInput"] textarea {
  font-size: 1rem;
}


/* User messages */
.user-message-row {
  display: flex;
  justify-content: flex-end;
  margin: 0.6rem 0 1rem;
}
.user-message-bubble {
  max-width: 74%;
  padding: 0.75rem 1rem;
  border-radius: 18px 18px 4px 18px;
  background: #2563eb;
  color: white;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.assistant-response {
  margin: 0.75rem 0 1.6rem;
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
  gap: 0.15rem !important;
  min-height: 2rem !important;
  align-items: center !important;
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
  display: block !important;          /* not flex — flex can ignore ellipsis */
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

/* Active (primary) chat title */
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[kind="primary"],
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[data-testid="baseButton-primary"] {

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
}

[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[kind="primary"] *,
[data-testid="stSidebar"]
[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
[data-testid="stButton"] button[data-testid="baseButton-primary"] * {

    display: block !important;
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;

    line-height: 2rem !important;
}

/* ⋮ menu stays compact and aligned */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-history-marker)
  [data-testid="stPopover"] {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  min-width: 2rem !important;
  margin: 0 !important;
  padding: 0 !important;
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
  [data-testid="stPopover"] button p {
  margin: 0 !important;
  text-align: center !important;
}


/* Three-dot chat menu — only ⋮, hide dropdown caret (fallback outside history) */
[data-testid="stSidebar"] [data-testid="stPopover"] button {
  min-height: 2rem !important;
  min-width: 2rem !important;
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

STAGE_PREFIX = {
    "clarifying": "… ",
    "plan_review": "Plan ",
    "spec_approved": "✓ ",
    "failed": "! ",
}


def create_welcome_message(chat_id):
    mode = "mock mode" if USE_MOCK else "connected to orchestration API"
    add_message(
        chat_id,
        "assistant",
        (
            "Welcome to the Workflow Assistant. Describe an orchestration in "
            "natural language. I may ask a clarifying question, then show a "
            "plan (summary + diagram + placeholders). You can approve the plan "
            f"or request changes. ({mode}.)\n\n"
            "ASL generation and deploy are not available until those APIs exist."
        ),
    )


def ensure_active_chat():
    active_id = st.session_state.get("active_chat_id")
    if active_id is not None and get_chat(active_id) is not None:
        return
    chats = get_chats()
    if chats:
        st.session_state.active_chat_id = chats[0]["id"]
        return
    chat_id = create_chat()
    create_welcome_message(chat_id)
    st.session_state.active_chat_id = chat_id


def is_draft_chat(chat_id):
    if get_chat(chat_id) is None:
        return False
    messages = get_messages(chat_id)
    return not any(m["role"] == "user" for m in messages)


def get_active_session_id(chat_id):
    """Backend session_id stored as chats.run_id."""
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
    active_id = st.session_state.get("active_chat_id")
    if active_id and get_chat(active_id) is not None and is_draft_chat(active_id):
        st.session_state.active_chat_id = active_id
        st.session_state.focus_chat_input = True
        return
    chat_id = create_chat()
    create_welcome_message(chat_id)
    st.session_state.active_chat_id = chat_id
    st.session_state.focus_chat_input = True


def create_chat_title(prompt):
    title = " ".join(prompt.strip().split())
    if not title:
        return "Untitled workflow"
    if len(title) > 45:
        return title[:45].rstrip() + "..."
    return title


def delete_chat_and_select_next(chat_id):
    delete_chat(chat_id)
    chats = get_chats()
    if chats:
        st.session_state.active_chat_id = chats[0]["id"]
    else:
        create_new_chat()


def render_sidebar_chat(chat):
    chat_id = chat["id"]
    is_active = chat_id == st.session_state.active_chat_id

    # Keep title to one visual line even if CSS fails
    label = chat["title"]
    if len(label) > 36:
        label = label[:36].rstrip() + "…"
        

    chat_column, option_column = st.columns(
        [0.90, 0.10],
        gap="small",          # tighter than "small"
        vertical_alignment="center",
    )

    with chat_column:
        if st.button(
            label,
            key=f"open_chat_{chat_id}",
            use_container_width=True,
            type="primary" if is_active else "tertiary",
        ):
            st.session_state.active_chat_id = chat_id
            st.session_state.focus_chat_input = True
            st.rerun()

    with option_column:
        with st.popover("⋮", help="Chat options", use_container_width=False):
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
                    st.rerun()

            with c2:
                if st.button(
                    "Delete",
                    key=f"delete_chat_{chat_id}",
                    use_container_width=True,
                ):
                    delete_chat_and_select_next(chat_id)
                    st.rerun()


def render_user_message(content):
    safe = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    st.markdown(
        f'<div class="user-message-row"><div class="user-message-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def apply_run_result(chat_id, result, user_text=None):
    # User message is already saved in queue_user_prompt (or reject button).
    # user_text is ignored to avoid duplicate user bubbles.
    _ = user_text

    assistant_text = result.get("assistant_text") or result.get("summary") or "Done."
    add_message(chat_id, "assistant", assistant_text, artifact=result)

    stage = result.get("stage")
    session_id = result.get("session_id") or result.get("run_id")

    deploy_status = {
        "clarifying": "clarifying",
        "plan_review": "plan_pending",
        "spec_approved": "approved",
        "failed": "failed",
    }.get(stage)

    update_chat_run_state(
        chat_id,
        run_id=session_id,
        stage=stage,
        deploy_status=deploy_status,
    )
    st.session_state.focus_chat_input = True


def render_clarifying(artifact):
    questions = artifact.get("clarifying_questions") or []
    if questions:
        st.markdown("**Clarifying question**")
        for q in questions:
            st.markdown(f"- {q}")
    st.info("Answer in the chat box below.")


def render_plan_review(message_id, artifact, chat_id):
    session_id = artifact.get("session_id") or artifact.get("run_id")

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
            )

    tabs = st.tabs(["Diagram", "Plan JSON", "Actions"])

    with tabs[0]:
        dot = artifact.get("graph_dot")
        if dot:
            st.graphviz_chart(dot)
        else:
            st.caption("No diagram from API.")

    with tabs[1]:
        plan = artifact.get("plan") or artifact.get("workflow_spec") or {}
        st.code(json.dumps(plan, indent=2), language="json")

    with tabs[2]:
        # Only show actions on the latest plan_review message
        messages = get_messages(chat_id)
        latest_plan_msg_id = None
        for m in reversed(messages):
            art = m.get("artifact") or {}
            if art.get("stage") == "plan_review":
                latest_plan_msg_id = m["id"]
                break
        is_latest = message_id == latest_plan_msg_id

        if not is_latest:
            st.caption("Older plan — use the latest plan card for actions.")
            return

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Approve plan",
                key=f"approve_plan_{message_id}",
                type="primary",
                use_container_width=True,
            ):
                clean = {k: v.strip() for k, v in overrides.items() if v and v.strip()}
                try:
                    with st.spinner("Approving plan…"):
                        result = approve_plan(session_id, overrides=clean)
                    apply_run_result(chat_id, result)
                    st.rerun()
                except OrchestrationApiError as exc:
                    st.error(str(exc))

        with c2:
            st.caption("Or type change requests in chat / use Reject below.")

        feedback = st.text_area(
            "Request changes",
            key=f"reject_fb_{message_id}",
            placeholder="e.g. Use Glue instead of Lambda; notify email on failure.",
        )
        if st.button(
            "Reject / request changes",
            key=f"reject_plan_{message_id}",
            use_container_width=True,
        ):
            if not (feedback or "").strip():
                st.warning("Enter feedback first.")
            else:
                # Same chat UX: show user bubble immediately, API on next run
                queue_user_prompt(chat_id, feedback.strip())


def render_spec_approved(artifact):
    st.success("Plan approved and locked on the server.")
    remaining = artifact.get("remaining_placeholders") or artifact.get("placeholders") or []
    if remaining:
        st.warning("Unresolved placeholders:")
        for p in remaining:
            st.markdown(f"- `{p}`")
    plan = artifact.get("plan") or artifact.get("workflow_spec")
    if plan:
        with st.expander("Resolved plan", expanded=True):
            st.code(json.dumps(plan, indent=2), language="json")
    st.info("Next steps (ASL / deploy) will appear when those APIs are available.")


def render_artifact(message_id, artifact, chat_id):
    if not artifact:
        return
    stage = artifact.get("stage") or "unknown"
    sid = artifact.get("session_id") or artifact.get("run_id") or "—"
    st.caption(f"Session `{sid}` · stage **{stage}**")

    if stage == "clarifying":
        render_clarifying(artifact)
    elif stage == "plan_review":
        render_plan_review(message_id, artifact, chat_id)
    elif stage == "spec_approved":
        render_spec_approved(artifact)
    elif stage == "failed":
        st.error(artifact.get("assistant_text") or "Something failed.")
    else:
        st.json(artifact)

def queue_user_prompt(active_chat_id, user_prompt):
    """Save user message immediately, then rerun so the bubble shows before the API call."""
    if get_chat(active_chat_id) is None:
        st.session_state.pop("active_chat_id", None)
        ensure_active_chat()
        active_chat_id = st.session_state.active_chat_id

    chat = get_chat(active_chat_id)
    if chat and chat.get("title") in ("New workflow chat", "Untitled workflow"):
        update_chat_title(active_chat_id, create_chat_title(user_prompt))

    add_message(active_chat_id, "user", user_prompt)
    st.session_state.pending_prompt = user_prompt
    st.session_state.pending_chat_id = active_chat_id
    st.session_state.focus_chat_input = True
    st.rerun()


def process_user_prompt(active_chat_id, user_prompt, region, env):
    """Call API and save assistant reply. User message must already be in DB."""
    if get_chat(active_chat_id) is None:
        st.session_state.pop("active_chat_id", None)
        ensure_active_chat()
        active_chat_id = st.session_state.active_chat_id

    chat = get_chat(active_chat_id)
    session_id = get_active_session_id(active_chat_id)

    # If user types while latest stage is plan_review, treat as reject feedback
    if chat and chat.get("stage") == "plan_review" and session_id and not USE_MOCK:
        try:
            with st.status("Sending change request…", expanded=True) as status:
                result = reject_plan(session_id, user_prompt, region=region, env=env)
                status.update(label=f"Stage: {result.get('stage')}", state="complete")
            apply_run_result(active_chat_id, result)
            st.session_state.pending_prompt = None
            st.session_state.pending_chat_id = None
            st.rerun()
            return
        except OrchestrationApiError as exc:
            st.error(str(exc))
            st.session_state.pending_prompt = None
            st.session_state.pending_chat_id = None
            return

    try:
        with st.status("Talking to orchestration API…", expanded=True) as status:
            st.write("Sending message…")
            result = send_message(
                user_prompt,
                run_id=session_id,
                region=region,
                env=env,
            )
            status.update(label=f"Stage: {result.get('stage')}", state="complete")
        apply_run_result(active_chat_id, result)
        st.session_state.pending_prompt = None
        st.session_state.pending_chat_id = None
        st.rerun()
    except OrchestrationApiError as exc:
        st.error(str(exc))
        st.session_state.pending_prompt = None
        st.session_state.pending_chat_id = None


def focus_chat_input(active_chat_id):
    st.session_state.pop("focus_chat_input", None)
    if st.session_state.get("_last_focused_chat") != active_chat_id:
        st.session_state._focus_tick = st.session_state.get("_focus_tick", 0) + 1
        st.session_state._last_focused_chat = active_chat_id
    nonce = f"{active_chat_id}-{st.session_state.get('_focus_tick', 0)}"
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          let attempts = 0;
          function tryFocus() {{
            const root = doc.querySelector('[data-testid="stChatInput"]');
            const input = root && (root.querySelector("textarea") || root.querySelector("input"));
            if (input) {{ input.focus(); return; }}
            attempts += 1;
            if (attempts < 40) setTimeout(tryFocus, 50);
          }}
          setTimeout(tryFocus, 0);
        }})();
        </script>
        """,
        height=0,
    )


# ----- body -----
ensure_active_chat()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # str | None
if "pending_chat_id" not in st.session_state:
    st.session_state.pending_chat_id = None

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

    with st.container(height=400,border=False):
        st.markdown(
            '<div class="sidebar-history-marker" style="display:none"></div>',
            unsafe_allow_html=True,
        )

        for chat in get_chats():
            if is_draft_chat(chat["id"]):
                continue
            render_sidebar_chat(chat)
        # Space below the final history item
        st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)

active_chat_id = st.session_state.active_chat_id
if not get_chat(active_chat_id):
    st.session_state.pop("active_chat_id", None)
    ensure_active_chat()
    active_chat_id = st.session_state.active_chat_id

st.title("Workflow Assistant")
st.caption("Chat → clarify → plan review → approve / reject")

messages = get_messages(active_chat_id)
has_user = any(m["role"] == "user" for m in messages)

for message in messages:
    if message["role"] == "user":
        render_user_message(message["content"])
    else:
        st.markdown('<div class="assistant-response">', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("artifact"):
            render_artifact(message["id"], message["artifact"], active_chat_id)
        st.markdown("</div>", unsafe_allow_html=True)

# Pending API work: user bubble already rendered above
pending = st.session_state.get("pending_prompt")
pending_cid = st.session_state.get("pending_chat_id")
if pending and pending_cid == active_chat_id:
    process_user_prompt(
        active_chat_id,
        pending,
        region=st.session_state.get("aws_region", "ap-south-1"),
        env=st.session_state.get("deploy_env", "dev"),
    )

if not has_user:
    st.markdown("##### Try an example")
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            if st.session_state.get("pending_prompt"):
                st.warning("Please wait for the current reply…")
            else:
                queue_user_prompt(active_chat_id, example)

user_prompt = st.chat_input("Describe a workflow, or answer the question…")
focus_chat_input(active_chat_id)

if user_prompt:
    if st.session_state.get("pending_prompt"):
        st.warning("Please wait for the current reply…")
    else:
        queue_user_prompt(active_chat_id, user_prompt)