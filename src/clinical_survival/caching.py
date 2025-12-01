from pathlib import Path
from joblib import Memory
from rich.console import Console

from clinical_survival.config import CachingConfig

console = Console()

# Global cache object
_cache_memory = None


def get_caching_memory(config: CachingConfig) -> Memory:
    """
    Initializes and returns a joblib.Memory object for caching.
    
    This function creates a singleton Memory object based on the provided
    caching configuration.
    """
    global _cache_memory
    
    if _cache_memory is None:
        if config.enabled:
            cache_dir = Path(config.dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"💾 Caching enabled. Using directory: [cyan]{cache_dir}[/cyan]")
            _cache_memory = Memory(location=cache_dir, verbose=0)
        else:
            console.print("💾 Caching is disabled.", style="yellow")
            # Use a "null" cacher that doesn't actually cache
            _cache_memory = Memory(location=None, verbose=0)
            
    return _cache_memory


