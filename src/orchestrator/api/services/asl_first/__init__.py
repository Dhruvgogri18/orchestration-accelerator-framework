"""ASL-first pipeline (PIPELINE_MODE=asl_first).

Flow:
  1. draft      -- LLM writes ASL directly from the conversation. Stored on
                   locked_spec["asl_draft"]; never shown to the user.
  2. plan view  -- a Plan is derived from the draft IN CODE (plan_view.py),
                   shown to the user for approval.
  3. approve    -- the LLM's clarifying questions about the draft are asked
                   (reuses the enrichment question/answer endpoint + phase).
  4. placeholders -- remaining TODO_ tokens are asked (reuses
                   /spec/placeholders) and patched into the ASL in code.
  5. final ASL  -- validated (incl. AWS ValidateStateMachineDefinition) and
                   shown. locked_spec["asl"] is only set at this point, so
                   derive_phase() keeps the user out of ASL review until
                   everything is resolved.
"""
