import sys
from typing import Dict, Type, Any, Iterator

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


class PluginRegistry:
    """A registry for discovering and managing plugins."""

    def __init__(self, group: str):
        self.group = group
        self._plugins: Dict[str, Type[Any]] = {}
        self._loaded = False

    def _load(self):
        """Load plugins from entry points."""
        if not self._loaded:
            discovered_plugins = entry_points(group=self.group)
            for plugin in discovered_plugins:
                self._plugins[plugin.name] = plugin.load()
            self._loaded = True

    def get(self, name: str) -> Type[Any]:
        """Get a plugin by name."""
        self._load()
        if name not in self._plugins:
            raise KeyError(
                f"Plugin '{name}' not found in group '{self.group}'. "
                f"Available plugins: {list(self._plugins.keys())}"
            )
        return self._plugins[name]

    def __iter__(self) -> Iterator[str]:
        self._load()
        return iter(self._plugins)

    def __len__(self) -> int:
        self._load()
        return len(self._plugins)


# Create registries for models and preprocessors
model_registry = PluginRegistry("clinical_survival.models")
preprocessor_registry = PluginRegistry("clinical_survival.preprocessors")
