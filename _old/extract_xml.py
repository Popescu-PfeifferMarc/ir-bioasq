import xml.etree.ElementTree as ET
import re
from typing import Optional, TypedDict

class ArticleData(TypedDict):
    pmid: Optional[str]
    title: Optional[str]
    body: Optional[str]

def extract_text(element: Optional[ET.Element])-> Optional[str]:
    if element is None:
        return None
    
    text_parts = []
    
    def recursive_extract(e):
        if e.text:
            text_parts.append(e.text)
        for child in e:
            recursive_extract(child)
            if child.tail:
                text_parts.append(child.tail)
        text_parts.append(" ")

    recursive_extract(element)
    
    combined_text = "".join(text_parts)
    combined_text = combined_text.replace("\t", " ")
    combined_text = re.sub(r'\s+', ' ', combined_text)
    combined_text = re.sub(r'\n+', '\n', combined_text)
    
    return combined_text.strip()

def extract_xml(path: str) -> ArticleData:
    tree = ET.parse(path)
    root = tree.getroot()
    
    pmid = extract_text(root.find("front/article-meta/article-id[@pub-id-type='pmid']"))
    title = extract_text(root.find("front/article-meta/title-group/article-title"))
    body = extract_text(root.find("body"))

    return {
        "pmid": pmid,
        "title": title,
        "body": body
    }
