"""Maneuvers and scripts given as source text, loaded into a running observer.

Each source becomes a fresh module rather than a reload of an old one, so nothing from a
previous version survives except what the observer still holds, and remove_maneuver lets go
of that. Sources are registered with linecache, so tracebacks show the agent's own lines.
"""

from __future__ import annotations

import inspect
import itertools
import linecache
import logging
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from nf_robot.host.maneuver import Maneuver

logger = logging.getLogger(__name__)

# module name prefixes; the bridge reports INFO logs from these, not just warnings
MANEUVER_PREFIX = 'agent_maneuvers'
SCRIPT_PREFIX = 'agent_scripts'
MODULE_NAME = re.compile(r'^[a-z_][a-z0-9_]*$')


@dataclass
class AgentModule:
    name: str
    source: str
    maneuver_names: list = field(default_factory=list)
    saved_path: Path | None = None


def compile_module(full_name, source):
    """Execute source as a new module named full_name and return it. A SyntaxError or anything
    the module raises at import propagates, and leaves no module behind."""
    filename = f'<{full_name}>'
    code = compile(source, filename, 'exec')
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(full_name)
    module.__file__ = filename
    # dataclasses and pickle look a class's module up by name
    previous = sys.modules.get(full_name)
    sys.modules[full_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(full_name, None)
        else:
            sys.modules[full_name] = previous
        raise
    return module


def maneuver_classes(module):
    """The Maneuver subclasses defined in module itself, not imported into it."""
    return [v for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, Maneuver) and v is not Maneuver
            and v.__module__ == module.__name__]


class AgentCode:
    """The maneuver modules an agent has loaded, and the folder they are saved to."""

    def __init__(self, ob, directory=None):
        self.ob = ob
        self.directory = Path(directory) if directory else None
        self.modules = {}
        self._script_ids = itertools.count(1)

    def load_directory(self):
        """Register every *.py in the directory with an observer that has not started yet.
        A file that fails is logged and skipped."""
        if self.directory is None or not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob('*.py')):
            try:
                self._register(path.stem, path.read_text(), running=False)
                self.modules[path.stem].saved_path = path
            except Exception:
                logger.exception(f'Could not load agent maneuver {path}')

    async def load(self, name, source):
        """Load or replace a maneuver module in the running observer. Returns its AgentModule.

        The new source is executed before the old version is removed, so a module that does
        not even import leaves the old one running."""
        if not MODULE_NAME.match(name):
            raise ValueError(f'module name {name!r} must be lower_snake_case')
        module = compile_module(f'{MANEUVER_PREFIX}.{name}', source)
        classes = maneuver_classes(module)
        if not classes:
            raise ValueError('the source defines no Maneuver subclass')
        await self.unload(name, missing_ok=True)
        entry = AgentModule(name, source)
        self.modules[name] = entry
        for cls in classes:
            maneuver = await self.ob.add_running_maneuver(cls)
            entry.maneuver_names.append(maneuver.name)
        return entry

    def _register(self, name, source, running):
        module = compile_module(f'{MANEUVER_PREFIX}.{name}', source)
        entry = AgentModule(name, source)
        self.modules[name] = entry
        for cls in maneuver_classes(module):
            entry.maneuver_names.append(self.ob.add_maneuver(cls).name)

    async def unload(self, name, missing_ok=False):
        entry = self.modules.pop(name, None)
        if entry is None:
            if missing_ok:
                return
            raise KeyError(f'no agent module named {name!r}; loaded: {sorted(self.modules)}')
        for maneuver_name in entry.maneuver_names:
            if maneuver_name in self.ob.maneuvers:
                await self.ob.remove_maneuver(maneuver_name)

    def save(self, name):
        """Write a loaded module's source into the directory, where the next start loads it."""
        if self.directory is None:
            raise ValueError('no maneuver directory; start stringman-headless with --maneuver_dir')
        entry = self.modules[name]
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{name}.py'
        path.write_text(entry.source)
        entry.saved_path = path
        return path

    def script(self, source):
        """The `run` coroutine function a script defines, from a module of its own."""
        module = compile_module(f'{SCRIPT_PREFIX}.script_{next(self._script_ids)}', source)
        run = getattr(module, 'run', None)
        if not inspect.iscoroutinefunction(run):
            raise ValueError('a script must define `async def run(ob, m)`')
        return run
