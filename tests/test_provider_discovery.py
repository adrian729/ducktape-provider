import unittest
from unittest.mock import MagicMock, patch

from ducktape_provider import ClaudeAdapter, Provider


class FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


class FakeAdapter:
    def is_available(self):
        return True

    def models(self):
        return []


class RaisingEntryPoint:
    def __init__(self, name):
        self.name = name

    def load(self):
        raise ImportError(f"no module for {self.name}")


class TestProviderDiscovery(unittest.TestCase):
    @patch("ducktape_provider.provider.importlib.metadata.entry_points")
    def test_autodiscover_false_never_calls_entry_points(self, mock_entry_points):
        provider = Provider()
        mock_entry_points.assert_not_called()
        self.assertEqual(
            set(provider.providers().keys()), {"claude", "openai", "ollama-local"}
        )

    @patch("ducktape_provider.provider.importlib.metadata.entry_points")
    def test_autodiscover_adds_novel_adapter(self, mock_entry_points):
        mock_entry_points.return_value = [FakeEntryPoint("myvendor", FakeAdapter)]
        provider = Provider(autodiscover=True)
        self.assertIn("myvendor", provider.providers())
        self.assertTrue(provider.providers()["myvendor"])

    @patch("ducktape_provider.provider.importlib.metadata.entry_points")
    def test_autodiscover_builtin_name_collision_keeps_builtin(self, mock_entry_points):
        mock_entry_points.return_value = [FakeEntryPoint("claude", FakeAdapter)]
        provider = Provider(autodiscover=True)
        self.assertIsInstance(provider._adapters["claude"], ClaudeAdapter)

    @patch("ducktape_provider.provider.importlib.metadata.entry_points")
    def test_autodiscover_with_explicit_adapters_precedence(self, mock_entry_points):
        explicit_claude = MagicMock()
        mock_entry_points.return_value = [
            FakeEntryPoint("claude", FakeAdapter),
            FakeEntryPoint("myvendor", FakeAdapter),
        ]
        provider = Provider(adapters={"claude": explicit_claude}, autodiscover=True)
        self.assertIs(provider._adapters["claude"], explicit_claude)
        self.assertIn("myvendor", provider._adapters)

    @patch("ducktape_provider.provider.importlib.metadata.entry_points")
    def test_autodiscover_skips_broken_entry_point(self, mock_entry_points):
        mock_entry_points.return_value = [
            RaisingEntryPoint("broken"),
            FakeEntryPoint("goodvendor", FakeAdapter),
        ]
        provider = Provider(autodiscover=True)
        self.assertNotIn("broken", provider._adapters)
        self.assertIn("goodvendor", provider._adapters)


if __name__ == "__main__":
    unittest.main()
