from dataclasses import dataclass
from pathlib import Path

from pydantic_monty import MountDir


@dataclass(frozen=True)
class WorkspaceConfig:
    """Configure one persistent host-backed virtual workspace.

    Attributes:
        path: Dedicated host directory that stores workspace files.
        virtual_path: Absolute POSIX path exposed inside Monty.
        write_bytes_limit: Maximum bytes Monty can write in one operation.
        memory_usage_limit: Maximum memory used by the mount.
    """

    path: str | Path
    virtual_path: str = "/workspace"
    write_bytes_limit: int = 10_000_000
    memory_usage_limit: int = 100_000_000

    def __post_init__(self) -> None:
        """Validate workspace paths and limits."""
        if not self.virtual_path.startswith("/") or self.virtual_path == "/":
            raise ValueError("virtual_path must be an absolute path other than /")
        if self.write_bytes_limit <= 0 or self.memory_usage_limit <= 0:
            raise ValueError("workspace limits must be greater than 0")

    def create_mount(self) -> MountDir:
        """Create the persistent read-write mount.

        Returns:
            Unopened Monty mount owned by the kernel.
        """
        host_path = Path(self.path)
        host_path.mkdir(parents=True, exist_ok=True)
        return MountDir(
            host_path=host_path,
            virtual_path=self.virtual_path,
            mode="read-write",
            write_bytes_limit=self.write_bytes_limit,
            memory_usage_limit=self.memory_usage_limit,
        )
