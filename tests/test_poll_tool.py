"""Tests for poll_until_done tool helper functions and logic."""

import pytest
from agents.tool_context import ToolContext
from agents.usage import Usage

from api_agent.agent.rest_agent import _get_nested_value, _set_nested_value


def _make_tool_ctx(tool_name: str = "poll_until_done", input_json: str = "") -> ToolContext:
    """Create a minimal ToolContext for testing tool invocation."""
    return ToolContext(
        context=None,
        usage=Usage(),
        tool_name=tool_name,
        tool_call_id="test_call",
        tool_arguments=input_json,
    )


class TestGetNestedValue:
    """Test dot-notation value extraction."""

    def test_simple_key(self):
        data = {"foo": "bar"}
        assert _get_nested_value(data, "foo") == "bar"

    def test_nested_key(self):
        data = {"polling": {"completed": True}}
        assert _get_nested_value(data, "polling.completed") is True

    def test_deep_nested(self):
        data = {"a": {"b": {"c": {"d": 42}}}}
        assert _get_nested_value(data, "a.b.c.d") == 42

    def test_missing_key_returns_none(self):
        data = {"foo": "bar"}
        assert _get_nested_value(data, "missing") is None

    def test_missing_nested_returns_none(self):
        data = {"foo": {"bar": 1}}
        assert _get_nested_value(data, "foo.missing.deep") is None

    def test_empty_path_returns_none(self):
        data = {"foo": "bar"}
        assert _get_nested_value(data, "") is None

    def test_none_data_returns_none(self):
        assert _get_nested_value(None, "foo") is None

    def test_array_index(self):
        data = {"trips": [{"id": 1}, {"id": 2}]}
        assert _get_nested_value(data, "trips.0.id") == 1
        assert _get_nested_value(data, "trips.1.id") == 2

    def test_array_index_out_of_bounds(self):
        data = {"trips": [{"id": 1}]}
        assert _get_nested_value(data, "trips.5.id") is None

    def test_array_nested_completion(self):
        """Real-world case: trips.0.isCompleted."""
        data = {"trips": [{"isCompleted": True, "results": []}]}
        assert _get_nested_value(data, "trips.0.isCompleted") is True


class TestSetNestedValue:
    """Test dot-notation value setting."""

    def test_simple_key(self):
        data = {"foo": "bar"}
        _set_nested_value(data, "foo", "baz")
        assert data["foo"] == "baz"

    def test_nested_key(self):
        data = {"polling": {"count": 1}}
        _set_nested_value(data, "polling.count", 2)
        assert data["polling"]["count"] == 2

    def test_creates_nested_structure(self):
        data = {}
        _set_nested_value(data, "a.b.c", 42)
        assert data["a"]["b"]["c"] == 42

    def test_empty_path_does_nothing(self):
        data = {"foo": "bar"}
        _set_nested_value(data, "", "baz")
        assert data == {"foo": "bar"}


class TestPollBlocking:
    """Test that poll_until_done respects allow_unsafe_paths."""

    @pytest.mark.asyncio
    async def test_post_blocked_without_whitelist(self):
        import json

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=(),  # No paths allowed
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        input_json = json.dumps(
            {
                "method": "POST",
                "path": "/search",
                "done_field": "polling.completed",
                "done_value": "true",
                "body": json.dumps({"polling": {"count": 1}}),
            }
        )
        result = await poll_tool.on_invoke_tool(_make_tool_ctx(input_json=input_json), input_json)
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "not allowed" in result_dict.get("error", "")

    @pytest.mark.asyncio
    async def test_post_allowed_with_whitelist(self):
        import json

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        input_json = json.dumps(
            {
                "method": "POST",
                "path": "/search/flights",
                "done_field": "polling.completed",
                "done_value": "true",
                "body": json.dumps({"polling": {"count": 1}}),
            }
        )
        result = await poll_tool.on_invoke_tool(_make_tool_ctx(input_json=input_json), input_json)
        result_dict = json.loads(result)
        # Will fail with connection error but NOT blocked
        assert "not allowed" not in result_dict.get("error", "")


class TestPollGuardrails:
    """Test guardrails prevent LLM mistakes."""

    @pytest.mark.asyncio
    async def test_done_field_not_found_returns_error(self):
        """If done_field doesn't exist in response, return error with available keys."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        # Mock response without the expected done_field
        mock_response = {"status": "pending", "results": []}

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            return_value={"success": True, "data": mock_response},
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "done_field": "polling.completed",  # doesn't exist
                    "done_value": "true",
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "not found" in result_dict["error"]
            assert "polling.completed" in result_dict["error"]
            # Should show available keys for debugging
            assert "status" in result_dict["error"] or "keys" in result_dict["error"]

    @pytest.mark.asyncio
    async def test_agent_specified_delay_ms(self):
        """Agent-specified delay_ms should override response delay."""
        import json
        import time
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        call_times = []

        async def mock_request(*args, **kwargs):
            call_times.append(time.time())
            # Server asks for 60s delay but agent specifies 100ms
            return {
                "success": True,
                "data": {"polling": {"completed": len(call_times) >= 2, "delayMs": 60000}},
            }

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "delay_ms": 100,  # Agent overrides to 100ms
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert len(call_times) == 2
            actual_delay = call_times[1] - call_times[0]
            assert actual_delay < 1.0  # 100ms + tolerance

    @pytest.mark.asyncio
    async def test_max_polls_error_shows_last_value(self):
        """max_polls exceeded should show last done_field value."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "data": {"polling": {"completed": False}},
            },
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "delay_ms": 1,
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "max_polls" in result_dict["error"].lower() or "exceeded" in result_dict["error"]
            # Should show what the actual value was
            assert "false" in result_dict["error"].lower() or "False" in result_dict["error"]

    @pytest.mark.asyncio
    async def test_auto_increment_polling_count(self):
        """polling.count in body should auto-increment between polls."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        received_bodies = []

        async def mock_request(*args, body=None, **kwargs):
            import copy

            received_bodies.append(copy.deepcopy(body))
            return {
                "success": True,
                "data": {"polling": {"completed": len(received_bodies) >= 3}},
            }

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/search/flights",
                    "body": json.dumps({"polling": {"count": 1}}),
                    "done_field": "polling.completed",
                    "done_value": "true",
                    "delay_ms": 1,
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            # Check counts incremented: 1, 2, 3
            assert received_bodies[0]["polling"]["count"] == 1
            assert received_bodies[1]["polling"]["count"] == 2
            assert received_bodies[2]["polling"]["count"] == 3

    @pytest.mark.asyncio
    async def test_numeric_done_field_zero_means_done(self):
        """retry.next == 0 should be detected as done (Flights API pattern)."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/flights/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # retry.next = 2, 1, 0 (0 means done)
            return {
                "success": True,
                "data": {"retry": {"next": max(0, 3 - call_count)}, "trips": []},
            }

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/flights/search",
                    "done_field": "retry.next",
                    "done_value": "0",  # 0 means done
                    "delay_ms": 1,
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert call_count == 3  # Should poll 3 times until retry.next == 0

    @pytest.mark.asyncio
    async def test_invalid_body_json_returns_error(self):
        """Invalid body JSON should return a friendly error."""
        import json

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/search/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        input_json = json.dumps(
            {
                "method": "POST",
                "path": "/search/flights",
                "done_field": "polling.completed",
                "done_value": "true",
                "body": "not-json",
            }
        )
        result = await poll_tool.on_invoke_tool(_make_tool_ctx(input_json=input_json), input_json)
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "invalid body json" in result_dict["error"].lower()

    @pytest.mark.asyncio
    async def test_works_without_polling_count_in_body(self):
        """Should work fine without polling.count in body."""
        import json
        from unittest.mock import AsyncMock, patch

        from api_agent.agent.rest_agent import _create_poll_tool
        from api_agent.context import RequestContext

        ctx = RequestContext(
            target_url="",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=("/status/*",),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )
        poll_tool = _create_poll_tool(ctx, "https://api.example.com")

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "success": True,
                "data": {"status": {"done": call_count >= 2}},
            }

        with patch(
            "api_agent.agent.rest_agent.execute_request",
            new_callable=AsyncMock,
            side_effect=mock_request,
        ):
            input_json = json.dumps(
                {
                    "method": "POST",
                    "path": "/status/check",
                    "body": json.dumps({"query": "test"}),  # No polling.count
                    "done_field": "status.done",
                    "done_value": "true",
                    "delay_ms": 1,
                }
            )
            result = await poll_tool.on_invoke_tool(
                _make_tool_ctx(input_json=input_json), input_json
            )
            result_dict = json.loads(result)
            assert result_dict["success"] is True
