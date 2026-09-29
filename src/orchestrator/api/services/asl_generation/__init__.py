"""LLM-based Stage 4 ASL generation (opt-in via ASL_GENERATION_MODE).

See llm_service.generate_asl_via_llm for the entry point and
asl_service.build_asl_from_spec for how/when it's dispatched to
instead of the deterministic translator.
"""