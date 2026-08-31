import base64
import mimetypes
from dataclasses import dataclass
from typing import Any

from .kernel import PersistentMontyKernel


@dataclass(frozen=True)
class MediaFile:
    """Hold one binary virtual file prepared for model input.

    Attributes:
        path: Absolute virtual file path.
        content: Original binary content.
        media_type: MIME type used for the model input.
    """

    path: str
    content: bytes
    media_type: str

    def as_data_url(self) -> str:
        """Encode the file as a Base64 data URL.

        Returns:
            Data URL containing the full file.
        """
        encoded = base64.b64encode(self.content).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"

    def as_langchain_image(self, detail: str | None = None) -> dict[str, Any]:
        """Build a LangChain OpenAI-style image content block.

        Args:
            detail: Optional provider image detail, such as ``low`` or ``high``.

        Returns:
            Image content block suitable for a ``HumanMessage`` content list.

        Raises:
            ValueError: If the file MIME type is not an image.
        """
        if not self.media_type.startswith("image/"):
            raise ValueError("the virtual file is not an image")
        image_url: dict[str, str] = {"url": self.as_data_url()}
        if detail is not None:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}


def read_media_file(
    kernel: PersistentMontyKernel,
    path: str,
    *,
    media_type: str | None = None,
    max_bytes: int | None = None,
) -> MediaFile:
    """Read a binary virtual file and identify its media type.

    Args:
        kernel: Kernel that owns the virtual filesystem.
        path: Absolute virtual POSIX file path.
        media_type: Optional MIME type override.
        max_bytes: Optional caller byte limit below the filesystem hard cap.

    Returns:
        Binary file with model input helpers.

    Raises:
        ValueError: If no MIME type can be determined.
        MontyError: If the sandbox denies or cannot read the file.
    """
    resolved_type = media_type or mimetypes.guess_type(path)[0]
    if resolved_type is None:
        raise ValueError("media_type is required for an unknown file extension")
    return MediaFile(
        path=path,
        content=kernel.download_file(path, max_bytes=max_bytes),
        media_type=resolved_type,
    )
