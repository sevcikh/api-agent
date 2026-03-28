"""Tests for dynamic tool naming middleware."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import NotFoundError, ValidationError
from fastmcp.server.middleware import MiddlewareContext
from mcp import types as mt
from mcp.types import Tool

from api_agent.context import RequestContext
from api_agent.middleware import DynamicToolNamingMiddleware, _get_tool_suffix, _inject_api_context
from api_agent.recipe.store import RecipeStore, sha256_hex


def _dummy_list_context() -> MiddlewareContext[mt.ListToolsRequest]:
    """Create a dummy MiddlewareContext for on_list_tools tests."""
    return cast(MiddlewareContext[mt.ListToolsRequest], object())


class TestGetToolSuffix:
    """Test internal tool name suffix extraction."""

    def test_underscore_prefix_query(self):
        assert _get_tool_suffix("_query") == "query"

    def test_underscore_prefix_execute(self):
        assert _get_tool_suffix("_execute") == "execute"

    def test_no_underscore_prefix(self):
        assert _get_tool_suffix("query") == "query"

    def test_double_underscore(self):
        assert _get_tool_suffix("__private") == "_private"


class TestInjectApiContext:
    """Test description injection with full hostname."""

    def test_rest_api_context(self):
        desc = "Ask questions in natural language."
        result = _inject_api_context(desc, "flights-api.example.com", "rest")
        assert result == "[flights-api.example.com REST API] Ask questions in natural language."

    def test_graphql_api_context(self):
        desc = "Query the API."
        result = _inject_api_context(desc, "catalog-graphql.example.com", "graphql")
        assert result == "[catalog-graphql.example.com GraphQL API] Query the API."

    def test_empty_description(self):
        result = _inject_api_context("", "api.example.com", "rest")
        assert result == "[api.example.com REST API] "


class TestToolTransformation:
    """Test tool name and description transformation."""

    def test_tool_name_with_prefix(self):
        """Verify tool names use prefix + suffix format."""
        prefix = "flights_api_example"
        internal_name = "_query"
        suffix = _get_tool_suffix(internal_name)
        expected = f"{prefix}_{suffix}"
        assert expected == "flights_api_example_query"
        assert len(expected) <= 32 + 6 + 1  # prefix(32) + suffix + underscore

    def test_description_includes_full_hostname(self):
        """Verify descriptions include full hostname."""
        hostname = "flights-api-qa.internal.example.com"
        result = _inject_api_context("Test.", hostname, "rest")
        assert hostname in result
        assert "REST API" in result


class TestBuildRecipeToolName:
    """Test recipe tool name construction and truncation."""

    def test_short_name(self):
        from api_agent.middleware import _build_recipe_tool_name

        assert _build_recipe_tool_name("get_users") == "r_get_users"

    def test_truncates_long_slug(self):
        from api_agent.middleware import MAX_TOOL_NAME_LEN, _build_recipe_tool_name

        slug = "a" * 100
        result = _build_recipe_tool_name(slug)
        assert len(result) <= MAX_TOOL_NAME_LEN
        assert result.startswith("r_")

    def test_exact_boundary(self):
        from api_agent.middleware import MAX_TOOL_NAME_LEN, _build_recipe_tool_name

        # slug that exactly fills MAX_TOOL_NAME_LEN - 2 (for "r_")
        slug = "x" * (MAX_TOOL_NAME_LEN - 2)
        result = _build_recipe_tool_name(slug)
        assert result == f"r_{slug}"
        assert len(result) == MAX_TOOL_NAME_LEN


class TestSchemaLoadGuards:
    """Test schema validation during tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_rest_schema_missing(self):
        middleware = DynamicToolNamingMiddleware()

        req_ctx = RequestContext(
            target_url="https://api.example.com/openapi.json",
            api_type="rest",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ("", "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with pytest.raises(RuntimeError, match="Failed to load OpenAPI schema"):
                        await middleware.on_list_tools(_dummy_list_context(), call_next)

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_graphql_schema_missing(self):
        middleware = DynamicToolNamingMiddleware()

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ("", "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with pytest.raises(RuntimeError, match="Failed to load GraphQL schema"):
                        await middleware.on_list_tools(_dummy_list_context(), call_next)


class TestRecipeToolListing:
    """Test recipe tool exposure and naming."""

    @pytest.mark.asyncio
    async def test_recipe_tool_uses_slug_without_api_prefix(self, monkeypatch):
        middleware = DynamicToolNamingMiddleware()
        store = RecipeStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.RECIPE_STORE", store)
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = {
            "tool_name": "list_users_reporting_to_manager",
            "params": {"manager_name": {"type": "str", "default": "Jane Doe"}},
            "steps": [
                {
                    "kind": "graphql",
                    "name": "users",
                    "query_template": "{ users { name } }",
                }
            ],
            "sql_steps": [],
        }
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users reporting to manager",
            recipe=recipe,
            tool_name=recipe["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return [Tool(name="_query", description="Query", inputSchema={"type": "object"})]

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        names = [t.name for t in tools]
        assert "r_list_users_reporting_to_manager" in names
        assert all(len(name) <= 60 for name in names)


class TestRecipeToolSchema:
    """Test recipe tool input schema."""

    @pytest.mark.asyncio
    async def test_all_params_required_even_with_defaults(self, monkeypatch):
        """Defaults are example values; all params must be explicitly provided."""
        middleware = DynamicToolNamingMiddleware()
        store = RecipeStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.RECIPE_STORE", store)
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = {
            "tool_name": "list_users",
            "params": {
                "user_id": {"type": "int", "default": 1},
                "active": {"type": "bool", "default": True},
            },
            "steps": [{"kind": "graphql", "name": "users", "query_template": "{ users { id } }"}],
            "sql_steps": [],
        }
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        tool = next(t for t in tools if t.name == "r_list_users")
        schema = tool.model_dump().get("parameters", {})
        # Params are top-level (flat), not nested under "params"
        assert "params" not in schema.get("properties", {})
        assert sorted(schema.get("required", [])) == ["active", "user_id"]
        # Defaults shown as description hints, not JSON Schema defaults
        assert "default" not in schema["properties"]["user_id"]
        assert "default" not in schema["properties"]["active"]
        # return_directly is optional top-level field
        assert "return_directly" in schema["properties"]

    @pytest.mark.asyncio
    async def test_default_none_has_no_description_hint(self, monkeypatch):
        """Params with default=None get no 'e.g.' hint; non-None defaults do."""
        middleware = DynamicToolNamingMiddleware()
        store = RecipeStore(max_size=10)
        monkeypatch.setattr("api_agent.middleware.RECIPE_STORE", store)
        monkeypatch.setattr("api_agent.middleware.settings.ENABLE_RECIPES", True)

        raw_schema = '{"schema":"ok"}'
        api_id = "graphql:https://api.example.com/graphql"
        schema_hash = sha256_hex(raw_schema)
        recipe = {
            "tool_name": "list_users",
            "params": {
                "user_id": {"type": "int", "default": None},
                "active": {"type": "bool", "default": True},
            },
            "steps": [{"kind": "graphql", "name": "users", "query_template": "{ users { id } }"}],
            "sql_steps": [],
        }
        store.save_recipe(
            api_id=api_id,
            schema_hash=schema_hash,
            question="List users",
            recipe=recipe,
            tool_name=recipe["tool_name"],
        )

        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return []

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = (raw_schema, "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    tools = await middleware.on_list_tools(_dummy_list_context(), call_next)

        tool = next(t for t in tools if t.name == "r_list_users")
        schema = tool.model_dump().get("parameters", {})
        # Params are top-level (flat)
        assert "params" not in schema.get("properties", {})
        # Both params required
        assert sorted(schema.get("required", [])) == ["active", "user_id"]
        # default=None -> "Required" only; default=True -> has "e.g." hint
        uid_props = schema["properties"]["user_id"]
        active_props = schema["properties"]["active"]
        assert "e.g." not in uid_props.get("description", "")
        assert "Required" in uid_props.get("description", "")
        assert "e.g." in active_props.get("description", "")


class TestRecipeToolErrors:
    """Ensure recipe tools signal errors via MCP exceptions."""

    @pytest.mark.asyncio
    async def test_recipe_tool_invalid_arguments_raises_validation_error(self):
        middleware = DynamicToolNamingMiddleware()
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return None

        # Non-dict arguments should raise ValidationError
        message = mt.CallToolRequestParams.model_construct(name="r_test", arguments="not_a_dict")
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                mock_headers.return_value = {
                    "x-target-url": req_ctx.target_url,
                    "x-api-type": req_ctx.api_type,
                }
                with pytest.raises(ValidationError, match="Invalid arguments"):
                    await middleware.on_call_tool(context, call_next)

    @pytest.mark.asyncio
    async def test_recipe_tool_not_found_raises_not_found(self):
        middleware = DynamicToolNamingMiddleware()
        req_ctx = RequestContext(
            target_url="https://api.example.com/graphql",
            api_type="graphql",
            target_headers={},
            allow_unsafe_paths=(),
            base_url=None,
            include_result=False,
            poll_paths=(),
        )

        async def call_next(_context):
            return None

        message = mt.CallToolRequestParams(name="r_missing", arguments={})
        context = MiddlewareContext(message=message)

        with patch("api_agent.middleware.get_http_headers") as mock_headers:
            with patch("api_agent.middleware.get_request_context", return_value=req_ctx):
                with patch(
                    "api_agent.middleware.load_schema_and_base_url",
                    new_callable=AsyncMock,
                ) as mock_fetch:
                    mock_fetch.return_value = ('{"schema":"ok"}', "")
                    mock_headers.return_value = {
                        "x-target-url": req_ctx.target_url,
                        "x-api-type": req_ctx.api_type,
                    }
                    with patch(
                        "api_agent.middleware.RECIPE_STORE.find_recipe_by_tool_slug",
                        return_value=None,
                    ):
                        with pytest.raises(NotFoundError, match="recipe not found"):
                            await middleware.on_call_tool(context, call_next)
