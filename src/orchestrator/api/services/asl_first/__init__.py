"""ASL-first pipeline (PIPELINE_MODE=asl_first).

Flow:
  1. draft      -- LLM writes ASL directly from the conversation. Stored on
                   locked_spec["asl_draft"]; never shown to the user.
  2. lock       -- Plan is derived in code (plan_view.py), then snapped to a
                   stable structure via plan matching + canonicalize. If the
                   topology drifted, one constrained realign call rewrites
                   the ASL to match (still far cheaper than the staged
                   multi-stage pipeline). Paraphrases of the same intent
                   therefore yield the same Mermaid plan.
  3. plan view  -- locked plan shown to the user for approval.
  4. approve    -- the LLM's clarifying questions about the draft are asked
                   (reuses the enrichment question/answer endpoint + phase).
  5. placeholders -- remaining TODO_ tokens are asked (reuses
                   /spec/placeholders) and patched into the ASL in code.
  6. final ASL  -- validated (incl. AWS ValidateStateMachineDefinition) and
                   shown. locked_spec["asl"] is only set at this point, so
                   derive_phase() keeps the user out of ASL review until
                   everything is resolved.
"""
