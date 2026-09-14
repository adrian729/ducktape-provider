import os
import unittest
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from ducktape_provider import Adapter, Provider
from ducktape_provider.provider import _clear_discovery_cache
from ducktape_provider.types import Message, Response, StreamEvent, ToolDef

ENTRY_POINTS = "ducktape_provider.provider.importlib.metadata.entry_points"
LOGGER = "ducktape_provider.provider"


class StubAdapter(Adapter):
    """Identifies itself through models() so tests can tell adapters apart publicly."""

    label = "stub"
    available = True

    def is_available(self) -> bool:
        return self.available

    def models(self) -> set[str]:
        return {self.label}

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        raise NotImplementedError

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        raise NotImplementedError


def adapter_class(label: str, available: bool = True) -> type[StubAdapter]:
    return type(
        f"Stub_{label}", (StubAdapter,), {"label": label, "available": available}
    )


class FakeEntryPoint:
    def __init__(self, name: str, target: Any, dist: str | None = "acme-llm"):
        self.name = name
        self.value = f"{dist}.module:{name}"
        self.dist = SimpleNamespace(name=dist) if dist is not None else None
        self._target = target
        self.load_calls = 0

    def load(self) -> Any:
        self.load_calls += 1
        if isinstance(self._target, BaseException):
            raise self._target
        return self._target


class TestProviderDiscovery(unittest.TestCase):
    def setUp(self):
        _clear_discovery_cache()
        self.addCleanup(_clear_discovery_cache)

    @patch(ENTRY_POINTS)
    def test_autodiscover_false_never_calls_entry_points(self, mock_entry_points):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:9"}):
            provider = Provider()
            names = set(provider.providers())
        mock_entry_points.assert_not_called()
        self.assertEqual(names, {"claude", "openai", "ollama-local"})

    @patch(ENTRY_POINTS)
    def test_autodiscover_adds_novel_adapter_under_plain_name(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("myvendor", adapter_class("mine"))
        ]
        provider = Provider(adapters={}, autodiscover=True)
        self.assertEqual(provider.models(), {"myvendor": ["mine"]})

    @patch(ENTRY_POINTS)
    def test_builtin_name_collision_registers_prefixed(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", adapter_class("plugin", available=False))
        ]
        env = {"ANTHROPIC_API_KEY": "key", "OLLAMA_HOST": "http://127.0.0.1:9"}
        with patch.dict(os.environ, env), self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(autodiscover=True)
            available = provider.providers()
        # The built-in keeps "claude" (available via env); the plugin reports False.
        self.assertTrue(available["claude"])
        self.assertFalse(available["acme-llm:claude"])
        self.assertIn("acme-llm", logs.output[0])
        self.assertIn("'acme-llm:claude'", logs.output[0])

    @patch(ENTRY_POINTS)
    def test_explicit_adapter_collision_registers_prefixed(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", adapter_class("plugin")),
            FakeEntryPoint("myvendor", adapter_class("mine")),
        ]
        explicit = {"claude": adapter_class("explicit")()}
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters=explicit, autodiscover=True)
        self.assertEqual(
            provider.models(),
            {
                "claude": ["explicit"],
                "acme-llm:claude": ["plugin"],
                "myvendor": ["mine"],
            },
        )

    @patch(ENTRY_POINTS)
    def test_same_name_across_dists_neither_gets_plain_name(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("shared", adapter_class("from-beta"), dist="beta-dist"),
            FakeEntryPoint("shared", adapter_class("from-alpha"), dist="alpha-dist"),
        ]
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(adapters={}, autodiscover=True)
        # Neither wins the plain name: which one would depend on dist sort
        # order, and a later install could silently reroute an existing
        # provider="shared" call to a different vendor.
        self.assertEqual(
            provider.models(),
            {"alpha-dist:shared": ["from-alpha"], "beta-dist:shared": ["from-beta"]},
        )
        self.assertIn("alpha-dist", logs.output[0])
        self.assertIn("beta-dist", logs.output[0])

    @patch(ENTRY_POINTS)
    def test_double_collision_skips_without_loading(self, mock_entry_points):
        entry_point = FakeEntryPoint("claude", adapter_class("plugin"))
        mock_entry_points.return_value = [entry_point]
        explicit = {
            "claude": adapter_class("explicit")(),
            "acme-llm:claude": adapter_class("explicit-prefixed")(),
        }
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(adapters=explicit, autodiscover=True)
        self.assertEqual(entry_point.load_calls, 0)
        self.assertEqual(
            provider.models(),
            {"claude": ["explicit"], "acme-llm:claude": ["explicit-prefixed"]},
        )
        self.assertIn("Skipping", logs.output[0])

    @patch(ENTRY_POINTS)
    def test_collision_without_distribution_is_skipped(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", adapter_class("plugin"), dist=None)
        ]
        explicit = {"claude": adapter_class("explicit")()}
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters=explicit, autodiscover=True)
        self.assertEqual(provider.models(), {"claude": ["explicit"]})

    @patch(ENTRY_POINTS)
    def test_allowlist_matches_plain_or_dist_qualified_names(self, mock_entry_points):
        wanted = FakeEntryPoint("wanted", adapter_class("wanted"))
        qualified = FakeEntryPoint("shared", adapter_class("beta"), dist="beta-dist")
        other_dist = FakeEntryPoint("shared", adapter_class("alpha"), dist="alpha")
        excluded = FakeEntryPoint("excluded", adapter_class("excluded"))
        mock_entry_points.return_value = [wanted, qualified, other_dist, excluded]

        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(
                adapters={},
                autodiscover={"wanted", "beta-dist:shared", "missing", "nope:shared"},
            )

        # "beta-dist:shared" was requested in qualified form, so it registers
        # under that exact name rather than the plain one.
        self.assertEqual(
            provider.models(),
            {"wanted": ["wanted"], "beta-dist:shared": ["beta"]},
        )
        self.assertEqual(excluded.load_calls, 0)
        self.assertEqual(other_dist.load_calls, 0)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("'missing', 'nope:shared'", logs.output[0])

    @patch(ENTRY_POINTS)
    def test_allowlist_without_typos_logs_nothing(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("wanted", adapter_class("wanted"))
        ]
        with self.assertNoLogs(LOGGER, "WARNING"):
            Provider(adapters={}, autodiscover={"wanted"})

    @patch(ENTRY_POINTS)
    def test_allowlist_generator_is_not_exhausted_by_validation(
        self, mock_entry_points
    ):
        # autodiscover is iterated once to type-check entries and again to build
        # the allowlist; a generator must survive both, not just the first.
        mock_entry_points.return_value = [
            FakeEntryPoint("wanted", adapter_class("wanted"))
        ]
        generator: Any = (name for name in ("wanted",))
        provider = Provider(adapters={}, autodiscover=generator)
        self.assertEqual(provider.models(), {"wanted": ["wanted"]})

    @patch(ENTRY_POINTS)
    def test_distribution_names_are_normalized(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", adapter_class("plugin"), dist="Acme_LLM.Tools"),
            FakeEntryPoint("other", adapter_class("other"), dist="Other_Dist"),
        ]
        explicit = {"claude": adapter_class("explicit")()}
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(
                adapters=explicit,
                autodiscover={"acme-llm-tools:claude", "OTHER.dist:other"},
            )
        # "other" was also requested in qualified form ("OTHER.dist:other"), so
        # it registers as "other-dist:other" even though nothing collides.
        self.assertEqual(
            provider.models(),
            {
                "claude": ["explicit"],
                "acme-llm-tools:claude": ["plugin"],
                "other-dist:other": ["other"],
            },
        )

    @patch(ENTRY_POINTS)
    def test_discovery_leaves_the_callers_adapters_mapping_alone(
        self, mock_entry_points
    ):
        mock_entry_points.return_value = [
            FakeEntryPoint("myvendor", adapter_class("mine"))
        ]
        explicit: dict[str, StubAdapter] = {"explicit": adapter_class("explicit")()}
        provider = Provider(adapters=explicit, autodiscover=True)
        self.assertEqual(set(provider.providers()), {"explicit", "myvendor"})
        self.assertEqual(set(explicit), {"explicit"})

    def test_allowlist_rejects_bare_string(self):
        with self.assertRaises(TypeError):
            Provider(adapters={}, autodiscover="myvendor")

    def test_allowlist_rejects_bytes(self):
        bogus: Any = b"myvendor"
        with self.assertRaises(TypeError):
            Provider(adapters={}, autodiscover=bogus)

    def test_allowlist_rejects_bytearray(self):
        bogus: Any = bytearray(b"myvendor")
        with self.assertRaises(TypeError):
            Provider(adapters={}, autodiscover=bogus)

    def test_allowlist_rejects_non_str_elements(self):
        bogus: Any = {"wanted", 5}
        with self.assertRaises(TypeError):
            Provider(adapters={}, autodiscover=bogus)

    @patch(ENTRY_POINTS)
    def test_two_same_name_plugins_get_no_plain_name(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("shared", adapter_class("from-yak"), dist="yak"),
            FakeEntryPoint("shared", adapter_class("from-zed"), dist="zed"),
        ]
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters={}, autodiscover=True)
        self.assertNotIn("shared", provider.models())
        self.assertEqual(
            provider.models(),
            {"yak:shared": ["from-yak"], "zed:shared": ["from-zed"]},
        )

    @patch(ENTRY_POINTS)
    def test_qualified_allowlist_entry_registers_under_qualified_name(
        self, mock_entry_points
    ):
        mock_entry_points.return_value = [
            FakeEntryPoint("tool", adapter_class("tool"), dist="vendor")
        ]
        with self.assertNoLogs(LOGGER, "WARNING"):
            provider = Provider(adapters={}, autodiscover={"vendor:tool"})
        # No collision at all, but the allowlist entry was qualified, so that
        # is the name registered — not the plain "tool".
        self.assertEqual(provider.models(), {"vendor:tool": ["tool"]})

    @patch(ENTRY_POINTS)
    def test_plain_allowlist_entry_colliding_with_builtin_registers_qualified(
        self, mock_entry_points
    ):
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", adapter_class("plugin"), dist="acme-llm")
        ]
        explicit = {"claude": adapter_class("explicit")()}
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters=explicit, autodiscover={"claude"})
        self.assertEqual(
            provider.models(),
            {"claude": ["explicit"], "acme-llm:claude": ["plugin"]},
        )

    @patch(ENTRY_POINTS)
    def test_ambiguous_plain_name_keyerror_names_qualified_alternatives(
        self, mock_entry_points
    ):
        mock_entry_points.return_value = [
            FakeEntryPoint("shared", adapter_class("from-yak"), dist="yak"),
            FakeEntryPoint("shared", adapter_class("from-zed"), dist="zed"),
        ]
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters={}, autodiscover=True)
        with self.assertRaisesRegex(
            KeyError, "yak:shared.*zed:shared|zed:shared.*yak:shared"
        ):
            provider.chat("model", [], provider="shared")

    @patch(ENTRY_POINTS)
    def test_non_adapter_entry_point_is_skipped(self, mock_entry_points):
        class NotAnAdapter:
            def is_available(self):
                return True

            def models(self):
                return set()

        mock_entry_points.return_value = [
            FakeEntryPoint("duck", NotAnAdapter),
            FakeEntryPoint("instance", adapter_class("instance")()),
            FakeEntryPoint("good", adapter_class("good")),
        ]
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(adapters={}, autodiscover=True)
        self.assertEqual(provider.models(), {"good": ["good"]})
        self.assertEqual(len(logs.output), 2)
        self.assertTrue(all("not an Adapter subclass" in line for line in logs.output))

    @patch(ENTRY_POINTS)
    def test_broken_entry_point_is_skipped_with_traceback(self, mock_entry_points):
        mock_entry_points.return_value = [
            FakeEntryPoint("broken", ImportError("no module"), dist="broken-dist"),
            FakeEntryPoint("goodvendor", adapter_class("good")),
        ]
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider = Provider(adapters={}, autodiscover=True)
        self.assertEqual(set(provider.providers()), {"goodvendor"})
        self.assertIn("broken-dist", logs.output[0])
        self.assertIsNotNone(logs.records[0].exc_info)

    @patch(ENTRY_POINTS)
    def test_system_exit_on_plugin_import_propagates(self, mock_entry_points):
        mock_entry_points.return_value = [FakeEntryPoint("x", SystemExit(3))]
        with self.assertRaises(SystemExit):
            Provider(adapters={}, autodiscover=True)

    @patch(ENTRY_POINTS)
    def test_system_exit_on_plugin_init_propagates(self, mock_entry_points):
        class ExitsOnInit(StubAdapter):
            def __init__(self):
                raise SystemExit(3)

        mock_entry_points.return_value = [FakeEntryPoint("x", ExitsOnInit)]
        with self.assertRaises(SystemExit):
            Provider(adapters={}, autodiscover=True)

    @patch(ENTRY_POINTS)
    def test_keyboard_interrupt_in_plugin_propagates(self, mock_entry_points):
        mock_entry_points.return_value = [FakeEntryPoint("x", KeyboardInterrupt())]
        with self.assertRaises(KeyboardInterrupt):
            Provider(adapters={}, autodiscover=True)

    @patch(ENTRY_POINTS)
    def test_failed_instantiation_of_contested_plugin_still_qualifies_survivor(
        self, mock_entry_points
    ):
        class FailsOnInit(StubAdapter):
            def __init__(self):
                raise RuntimeError("bad config")

        mock_entry_points.return_value = [
            FakeEntryPoint("shared", FailsOnInit, dist="alpha"),
            FakeEntryPoint("shared", adapter_class("beta"), dist="beta"),
        ]
        with self.assertLogs(LOGGER, "WARNING"):
            provider = Provider(adapters={}, autodiscover=True)
        # The plain name stays unclaimed even though the entry contesting it
        # failed to instantiate: the collision was about the two discovered
        # entry points sharing a name, decided before either was loaded.
        self.assertEqual(provider.models(), {"beta:shared": ["beta"]})

    @patch(ENTRY_POINTS)
    def test_discovery_is_cached_but_adapters_are_fresh(self, mock_entry_points):
        instances: list[StubAdapter] = []

        class Counted(StubAdapter):
            def __init__(self) -> None:
                instances.append(self)

        entry_point = FakeEntryPoint("myvendor", Counted)
        mock_entry_points.return_value = [entry_point]

        Provider(adapters={}, autodiscover=True)
        Provider(adapters={}, autodiscover=True)

        mock_entry_points.assert_called_once()
        self.assertEqual(entry_point.load_calls, 1)
        self.assertEqual(len(instances), 2)
        self.assertIsNot(instances[0], instances[1])

        _clear_discovery_cache()
        Provider(adapters={}, autodiscover=True)
        self.assertEqual(mock_entry_points.call_count, 2)
        self.assertEqual(entry_point.load_calls, 2)

    @patch(ENTRY_POINTS)
    def test_load_failure_is_retried_by_later_providers(self, mock_entry_points):
        entry_point = FakeEntryPoint("flaky", ImportError("mid-upgrade"))
        mock_entry_points.return_value = [entry_point]
        with self.assertLogs(LOGGER, "WARNING") as logs:
            first = Provider(adapters={}, autodiscover=True)
        self.assertEqual(first.models(), {})
        self.assertEqual(len(logs.output), 1)

        entry_point._target = adapter_class("recovered")
        second = Provider(adapters={}, autodiscover=True)
        third = Provider(adapters={}, autodiscover=True)
        self.assertEqual(second.models(), {"flaky": ["recovered"]})
        self.assertEqual(third.models(), {"flaky": ["recovered"]})
        # Failed once, then loaded once and cached.
        self.assertEqual(entry_point.load_calls, 2)


if __name__ == "__main__":
    unittest.main()
