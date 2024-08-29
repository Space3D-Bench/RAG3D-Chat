from pathlib import Path
from typing import List
import base64
from mimetypes import guess_type

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import ImageDocument


def load_text_documents(dir_path: Path) -> List[Document]:
    """
    Loads the textual data from files in a directory into LLamaIndex's
    Documents (each file being a separate document, its metadata's
    specifying the room name)

    Args:
        dir_path (Path): path to the directory with .txt files

    Returns:
        List[Document]: list of documents created from the files
    """
    docs: List[Document] = []
    for entry in dir_path.iterdir():
        if entry.name.endswith(".txt"):
            with entry.open("r") as file:
                content = file.read()
                docs.append(Document(text=content, metadata={"room_name": entry.stem}))

    return docs


def load_image_documents(dir_path: Path) -> List[Document]:
    """
    Loads the images (PNG, JPG, JPEG formats) from a specified directory
    into LLamaIndex's Documents; the directory contains subdirectories
    whose name indicate the corresponding room name

    Args:
        dir_path (Path): path to the directory with subdirectories
            containing corresponding images

    Returns:
        List[Document]: list of documents created from the images
    """
    docs = []
    for entry in dir_path.iterdir():
        if entry.is_dir():
            curr_docs = SimpleDirectoryReader(
                input_dir=dir_path / entry.name, required_exts=[".png", ".jpg", ".jpeg"]
            ).load_data()
            for i in range(len(curr_docs)):
                curr_docs[i].metadata = {
                    **curr_docs[i].metadata,
                    "room_name": entry.name,
                }
            docs.extend(curr_docs)

    return docs


def local_image_to_document(image_path: str) -> ImageDocument:
    """
    Creates an image document from a local image.

    Args:
        image_path (str): string path to a local image

        Returns:
            ImageDocument: document created from the local image
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")

    return ImageDocument(image=base64str, image_mimetype=mime_type)
