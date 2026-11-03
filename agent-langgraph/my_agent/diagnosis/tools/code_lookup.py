"""LLM tool for looking up class source code in a framework class index."""

from pathlib import Path
import logging

from my_agent.scripts.extract_forward_from_index import load_index, parse_python_file_for_class, resolve_torch_class_definition


# @tool("get_class_definition")
def get_class_definition(class_name: str, index_path: str) -> str:
    """Get the source code of a class from a class index
    
    Args:
        class_name (str): The name of the class.
        index_path (str): The path to the class index.
    """
    logging.info(f"[Tool][get_class_definition] Getting class definition for {class_name} from {index_path}")
    try:
        index = load_index(Path(index_path).resolve())
    except Exception as e:
        return f"Error loading index: {e}"
    if class_name in index:
        class_info = index[class_name][0]
        # print(class_info)
        class_node, src = parse_python_file_for_class(class_info.file, class_name)
        # logging.info(f"[Tool][get_class_definition] Class definition for {class_name}: {src}")
        return src
    else:
        src = resolve_torch_class_definition(class_name)["source"]
        # logging.info(f"[Tool][get_class_definition] Class definition for {class_name}: {src}")
        return src
