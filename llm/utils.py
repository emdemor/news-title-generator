import os
from pathlib import Path
from typing import Any, List, Union

import joblib


def join_sentences(sentences: List[str]) -> str:
    """
    Junta uma lista de frases em uma única string, separando as frases por um espaço.

    Args:
        sentences: Uma lista de frases.

    Returns:
        A string resultante da junção das frases, separadas por um espaço.
    """
    return " ".join(sentences)


def save_bin(obj: Any, path: str) -> None:
    """
    Salva um objeto binário em um arquivo em disco.

    Args:
        obj: O objeto a ser salvo.
        path: O caminho do arquivo em que o objeto será salvo.
    """
    create_folder_chain(path)
    joblib.dump(obj, path)


def load_bin(path: str) -> Any:
    """
    Carrega um objeto binário de um arquivo em disco.

    Args:
        path: O caminho do arquivo em que o objeto foi salvo.

    Returns:
        O objeto binário carregado do arquivo.
    """
    return joblib.load(path)


def create_folder_chain(path: Union[str, Path]) -> None:
    """
    Cria a cadeia de pastas necessária para um caminho de arquivo.

    Args:
        path: O caminho do arquivo ou da pasta a ser criada.
    """
    path_obj = Path(path)
    if path_obj.is_dir():
        os.makedirs(path_obj, exist_ok=True)
    else:
        parent_dir = path_obj.parent
        os.makedirs(parent_dir, exist_ok=True)
