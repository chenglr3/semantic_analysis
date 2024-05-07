import re
from typing import Union


def parse_type(text: str) -> Union[bool, str]:
    if text in ["true", "True"]:
        return True
    if text in ["false", "False"]:
        return False
    return text


def parse_output(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    for tag in tags:
        texts = re.findall(rf'<{tag}>(.*?)</{tag}>', output_str, re.DOTALL)

        if texts:
            xml_dict[tag] = parse_type(texts[0]) if first_item_only else texts
    return xml_dict