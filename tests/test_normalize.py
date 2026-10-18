"""Unit tests for the common request/response shape across providers."""

from __future__ import annotations

import base64
import json

from crashlab.capture import normalize as N

SYSTEM = "You are a refund agent."
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}
PNG = base64.b64encode(b"\x89PNG\r\n" + b"x" * 3000).decode()


def openai_conversation() -> dict:
    return {
        "model": "m",
        "temperature": 0,
        "max_completion_tokens": 256,
        "stop": "END",
        "tool_choice": {"type": "function", "function": {"name": "get_order"}},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": "Refund A17"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_id":  "A17"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"total": 49}'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Look up an order.",
                    "parameters": TOOL_SCHEMA,
                },
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_headers": {"X-Debug": "1"},
        "timeout": 10,
    }


def anthropic_conversation() -> dict:
    return {
        "model": "m",
        "temperature": 0,
        "max_tokens": 256,
        "stop_sequences": ["END"],
        "tool_choice": {"type": "tool", "name": "get_order"},
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "Refund A17"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "get_order",
                        "input": {"order_id": "A17"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": [{"type": "text", "text": '{"total": 49}'}],
                    }
                ],  # fmt: skip
            },
        ],
        "tools": [
            {
                "name": "get_order",
                "description": "Look up an order.",
                "input_schema": TOOL_SCHEMA,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def responses_conversation() -> dict:
    return {
        "model": "m",
        "temperature": 0,
        "max_output_tokens": 256,
        "stop": ["END"],
        "tool_choice": {"type": "function", "name": "get_order"},
        "instructions": SYSTEM,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Refund A17"}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_order",
                "arguments": '{"order_id":"A17"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": '{"total": 49}'},
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_order",
                "description": "Look up an order.",
                "parameters": TOOL_SCHEMA,
            }
        ],
    }


EXPECTED = {
    "system": SYSTEM,
    "messages": [
        {"role": "user", "parts": [{"type": "text", "content": "Refund A17"}]},
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "c1",
                    "name": "get_order",
                    "arguments": {"order_id": "A17"},
                }
            ],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "c1", "response": '{"total": 49}'}],
        },
    ],
    "tools": [
        {
            "type": "function",
            "name": "get_order",
            "description": "Look up an order.",
            "parameters": TOOL_SCHEMA,
        }
    ],
    "params": {
        "model": "m",
        "temperature": 0,
        "max_tokens": 256,
        "stop": ["END"],
        "tool_choice": {"type": "tool", "name": "get_order"},
    },
}


def test_openai_chat_request():
    assert N.normalize_openai_chat_request(openai_conversation()) == EXPECTED


def test_anthropic_request():
    assert N.normalize_anthropic_request(anthropic_conversation()) == EXPECTED


def test_responses_request():
    assert N.normalize_openai_responses_request(responses_conversation()) == EXPECTED


def test_equivalent_requests_hash_identically_across_providers():
    hashes = {
        N.request_hash(N.normalize_openai_chat_request(openai_conversation())),
        N.request_hash(N.normalize_anthropic_request(anthropic_conversation())),
        N.request_hash(N.normalize_openai_responses_request(responses_conversation())),
    }
    assert len(hashes) == 1


def test_hash_ignores_key_order_and_argument_whitespace():
    a = openai_conversation()
    b = json.loads(json.dumps(a))
    b["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{ "order_id" : "A17" }'
    b = {k: b[k] for k in reversed(list(b))}  # reverse top-level key order
    b["tools"][0]["function"] = dict(reversed(list(b["tools"][0]["function"].items())))
    assert N.request_hash(N.normalize_openai_chat_request(a)) == N.request_hash(
        N.normalize_openai_chat_request(b)
    )
    c = json.loads(json.dumps(a))
    c["messages"][1]["content"] = "Refund A18"
    assert N.request_hash(N.normalize_openai_chat_request(a)) != N.request_hash(
        N.normalize_openai_chat_request(c)
    )


def test_invalid_tool_arguments_kept_raw():
    msgs = N.normalize_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": '{"a": 1'}}],
            }
        ]
    )
    assert msgs[0]["parts"][0]["arguments"] == '{"a": 1'
    assert N.parse_arguments("") == {} and N.parse_arguments(None) == {}


def test_multipart_content_and_images_are_placeholders():
    data_url = f"data:image/png;base64,{PNG}"
    msgs = N.normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                    {"type": "input_audio", "input_audio": {"data": PNG, "format": "wav"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                    },
                    {"type": "image", "source": {"type": "url", "url": "https://x.test/b.png"}},
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": PNG},
                        "title": "inv.pdf",
                    },
                ],
            },  # fmt: skip
        ]
    )
    p1, p2 = msgs[0]["parts"], msgs[1]["parts"]
    assert p1[0] == {"type": "text", "content": "What is this?"}
    assert p1[1]["type"] == "image" and p1[1]["source"] == "base64"
    assert p1[1]["media_type"] == "image/png" and p1[1]["detail"] == "low"
    assert p1[1]["placeholder"].startswith("<image/png 3006 bytes sha256:")
    assert p1[2] == {"type": "image", "source": "url", "url": "https://x.test/a.png"}
    assert p1[3]["type"] == "audio" and "audio/wav" in p1[3]["placeholder"]
    # the same image via Anthropic yields the same placeholder
    assert p2[0]["placeholder"] == p1[1]["placeholder"]
    assert p2[1] == {"type": "image", "source": "url", "url": "https://x.test/b.png"}
    assert p2[2]["type"] == "file" and p2[2]["filename"] == "inv.pdf"
    assert PNG not in json.dumps(msgs)


def test_strip_binary_raw_payloads():
    raw = {"messages": [{"content": [{"image_url": {"url": f"data:image/png;base64,{PNG}"}}]}],
           "blob": PNG, "short": "abc", "text": "hello " * 400}  # fmt: skip
    out = N.strip_binary(raw)
    assert PNG not in json.dumps(out)
    assert out["short"] == "abc" and out["text"] == raw["text"]


def test_anthropic_tool_results_split_and_errors():
    msgs = N.normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "b",
                        "content": "boom",
                        "is_error": True,
                    },
                    {"type": "text", "text": "and also..."},
                ],
            }
        ]
    )
    assert msgs == [
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "a", "response": "ok"}]},
        {
            "role": "tool",
            "parts": [
                {"type": "tool_call_response", "id": "b", "response": "boom", "is_error": True}
            ],
        },
        {"role": "user", "parts": [{"type": "text", "content": "and also..."}]},
    ]


def test_thinking_and_unknown_blocks():
    msgs = N.normalize_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "xxx"},
                    {"type": "web_search_tool_result", "tool_use_id": "s", "content": []},
                ],
            }
        ]
    )
    parts = msgs[0]["parts"]
    assert parts[0] == {"type": "reasoning", "content": "hmm"}
    assert parts[1]["type"] == "reasoning"
    assert parts[2]["type"] == "web_search_tool_result" and parts[2]["data"]["tool_use_id"] == "s"


def test_developer_role_and_mid_conversation_system():
    req = N.normalize_openai_chat_request(
        {
            "model": "m",
            "messages": [
                {"role": "developer", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "C"},
            ],
        }
    )
    assert req["system"] == "A\n\nB"
    assert req["messages"][-1] == {"role": "system", "parts": [{"type": "text", "content": "C"}]}


def test_server_tools_and_tool_choice_variants():
    tools = N.normalize_tools(
        [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    )
    assert tools == [
        {"type": "web_search_20250305", "name": "web_search", "config": {"max_uses": 3}}
    ]
    assert N.normalize_tool_choice("auto") == {"type": "auto"}
    assert N.normalize_tool_choice({"type": "any"}) == {"type": "required"}
    assert N.normalize_tool_choice("required") == {"type": "required"}
    params = N.normalize_params(
        {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    )
    assert params == {"tool_choice": {"type": "auto"}, "parallel_tool_calls": False}


def test_litellm_transport_kwargs_dropped():
    req = N.normalize_openai_chat_request(
        {
            "model": "anthropic/claude",
            "messages": [],
            "api_key": "sk-x",
            "api_base": "http://h",
            "mock_response": "hi",
            "num_retries": 2,
        }  # fmt: skip
    )
    assert req["params"] == {"model": "anthropic/claude"}


# --- responses ---------------------------------------------------------------------


def test_openai_chat_response():
    norm, raw = N.normalize_openai_chat_response(
        {
            "id": "r1",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"x":1}'},
                            }
                        ],  # fmt: skip
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 4},
            },  # fmt: skip
        }
    )
    assert norm["tool_calls"] == [
        {"id": "c1", "type": "function", "name": "f", "arguments": {"x": 1}}
    ]
    assert norm["finish_reason"] == "tool_calls" and norm["provider_finish_reason"] == "tool_calls"
    assert norm["usage"] == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                             "cache_read_input_tokens": 4}  # fmt: skip
    assert norm["messages"] == [{"role": "assistant", "parts": [
        {"type": "tool_call", "id": "c1", "name": "f", "arguments": {"x": 1}}]}]  # fmt: skip
    assert raw["id"] == "r1"


def test_anthropic_response_matches_openai_shape():
    a_norm, _ = N.normalize_anthropic_response(
        {
            "id": "r1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "c1", "name": "f", "input": {"x": 1}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 6, "output_tokens": 3, "cache_read_input_tokens": 4},
        }
    )
    o_norm, _ = N.normalize_openai_chat_response(
        {
            "id": "r1",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Checking.",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"x": 1}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        }  # fmt: skip
    )
    a_norm.pop("provider_finish_reason")
    o_norm.pop("provider_finish_reason")
    assert a_norm == o_norm  # input tokens incl. cache reads, same parts, same finish reason


def test_anthropic_stop_reasons():
    for stop, expected in [("end_turn", "stop"), ("max_tokens", "length"),
                           ("stop_sequence", "stop"), ("refusal", "content_filter")]:  # fmt: skip
        norm, _ = N.normalize_anthropic_response(
            {"content": [], "stop_reason": stop, "usage": {"input_tokens": 1, "output_tokens": 1}}
        )
        assert norm["finish_reason"] == expected and norm["provider_finish_reason"] == stop


def test_responses_api_response():
    norm, _ = N.normalize_openai_responses_response(
        {
            "id": "resp_1",
            "model": "m",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            ],  # fmt: skip
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 1},
            },  # fmt: skip
        }
    )
    assert norm["messages"][0]["parts"] == [
        {"type": "reasoning", "content": "think"},
        {"type": "text", "content": "Hello"},
        {"type": "tool_call", "id": "c1", "name": "f", "arguments": {}},
    ]
    assert norm["finish_reason"] == "tool_calls"
    assert norm["usage"]["reasoning_tokens"] == 1
    inc, _ = N.normalize_openai_responses_response(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
    )
    assert inc["finish_reason"] == "length"
