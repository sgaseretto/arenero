"""Compare Monty features in an isolated interpreter without changing Arenero."""

import inspect
import json
from importlib.metadata import version
from typing import Any

import pydantic_monty as monty

CASES = {
    "eval": "eval('20 + 22')",
    "exec": "exec('answer = 42')\nanswer",
    "exec_namespace": "ns = {}\nexec('answer = 42', ns)\nns['answer']",
    "locals": "value = 42\nlocals()['value']",
    "compile": "compile('1 + 1', '<probe>', 'eval')",
    "globals": "globals()",
    "dynamic_import": "__import__('math')",
    "base64": "import base64\nbase64.b64encode(b'hello').decode()",
    "ascii85": "import base64\nbase64.a85decode(base64.a85encode(b'hello')).decode()",
    "binascii": "import binascii\nbinascii.crc_hqx(b'hello', 0)",
    "copy": "import copy\ncopy.deepcopy({'items': [1, 2]})",
    "random": "import random\nrandom.seed(42)\nrandom.randint(1, 100)",
    "time": "import time\ntime.process_time()",
    "asyncio_sleep": "import asyncio\nasyncio.run(asyncio.sleep(0, result=42))",
    "asyncio_create_task": "import asyncio\nasyncio.create_task",
    "complex": "repr((1 + 2j) * (1 - 2j))",
    "format": "format(42, '04d')",
    "percent_format": "'%s %d' % ('answer', 42)",
    "bytes_percent_format": "b'%s %d' % (b'answer', 42)",
    "datetime_now": "from datetime import datetime\nisinstance(datetime.now(), datetime)",
    "functools_reduce": "from functools import reduce\nreduce(lambda a, b: a + b, [20, 22])",
    "functools_partial": "from functools import partial\nf = partial(pow, 2)\nf(3)",
    "itertools_tee": "from itertools import tee\na, b = tee([1, 2])\n(list(a), list(b))",
    "itertools_batched": "from itertools import batched\nlist(batched([1, 2, 3], 2))",
    "iter_next": "it = iter([1, 2])\n(next(it), next(it))",
    "eager_builtins": "(type(map(str, [1])).__name__, type(enumerate([1])).__name__)",
    "eager_generator_expression": "type(x for x in [1, 2]).__name__",
    "class_iterator": (
        "class Items:\n    def __iter__(self):\n        return iter([20, 22])\n"
        "list(Items())"
    ),
    "dataclass_options": (
        "from dataclasses import dataclass\n@dataclass(frozen=True, eq=True)\n"
        "class Point:\n    x: int\nPoint(1) == Point(1)"
    ),
    "dataclass_field": (
        "from dataclasses import dataclass, field\n@dataclass\nclass Items:\n"
        "    values: list = field(default_factory=list)\nItems().values"
    ),
    "dataclass_post_init": (
        "from dataclasses import dataclass\n@dataclass\nclass Point:\n"
        "    x: int\n    def __post_init__(self):\n        self.x += 1\nPoint(1).x"
    ),
    "file_iteration": "[line for line in open('/probe.txt')]",
    "file_newline": "open('/probe.txt', newline='').read()",
    "file_update_mode": "open('/probe.txt', 'r+')",
    "file_exclusive_mode": "open('/new.txt', 'x')",
    "native_cwd": "import os\nos.getcwd()",
    "os_path": "import os.path\nos.path.join('/data', 'item.txt')",
    "posixpath": "import posixpath\nposixpath.basename('/data/item.txt')",
    "path_glob": "from pathlib import Path\n[str(p) for p in Path('/').glob('*.txt')]",
    "path_walk": "from pathlib import Path\nlist(Path('/').walk())",
    "os_scandir": "import os\n[e.name for e in os.scandir('/')]",
    "hashlib": "import hashlib\nhashlib.sha256(b'hello').hexdigest()",
    "print_stderr": "import sys\nprint('probe', file=sys.stderr)\nTrue",
    "third_party_import": "import numpy",
    "inheritance": "class Parent:\n    pass\nclass Child(Parent):\n    pass",
    "generator_function": "def values():\n    yield 1\nlist(values())",
    "match": "match 1:\n    case 1:\n        answer = 42\nanswer",
    "delete": "value = 1\ndel value",
}


def run_probe(pool: Any, source: str, limits: dict[str, Any]) -> dict[str, Any]:
    """Run one bounded feature check in a fresh Monty session.

    Args:
        pool: Active synchronous Monty worker pool.
        source: Monty source to submit. It never executes in host Python.
        limits: Limits supported by the installed Monty version.

    Returns:
        Success and value, or the exception type and message.
    """
    filesystem = monty.OSAccess(files=[monty.MemoryFile("/probe.txt", "one\r\ntwo\n")])
    try:
        with pool.checkout(limits=limits) as session:
            result = session.feed_run(
                source, os=filesystem, print_callback=lambda *_: None
            )
            return {"ok": True, "value": result}
    except (monty.MontyError, ValueError, TypeError) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)[:500]}


def main() -> None:
    """Print JSON evidence for the installed runtime and its public API.

    Returns:
        None. The report is written to standard output.
    """
    keys = sorted(monty.ResourceLimits.__annotations__)
    duration_key = (
        "max_feed_duration_secs"
        if "max_feed_duration_secs" in keys
        else "max_duration_secs"
    )
    limits = {
        duration_key: 1.0,
        "max_memory": 64 * 1024 * 1024,
        "max_recursion_depth": 100,
    }
    with monty.Monty(request_timeout=5) as pool:
        cases = {
            name: run_probe(pool, source, limits) for name, source in CASES.items()
        }
        legacy_limit = run_probe(pool, "1 + 1", {"max_duration_secs": 1.0})
    report = {
        "version": version("pydantic-monty"),
        "api": {
            "checkout": str(inspect.signature(monty.Monty.checkout)),
            "resource_limit_keys": keys,
            "host_objects": hasattr(monty, "ClassInstance"),
            "host_class_construction": hasattr(monty, "ClassType"),
            "sandbox_class_proxy": hasattr(monty, "MontyClassProxy"),
            "external_modules": hasattr(monty, "ExternalModule"),
        },
        "legacy_duration_limit": legacy_limit,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, default=repr, sort_keys=True))


if __name__ == "__main__":
    main()
