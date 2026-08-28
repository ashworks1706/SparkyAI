//! The agent loop: model call → policy → tool execution → repeat until final answer, error,
//! cancel, deadline, or step limit. Tests against mock model, tools, and policy live in this module.
