"""검색용 표현만 정리한다. 인용·응답용 공식 원문은 변경하지 않는다."""
from html import unescape
from html.parser import HTMLParser
import re


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "tr", "div"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in {"p", "tr", "div"}:
            self.parts.append("\n")


def clean_search_text(text):
    parser = _Text()
    parser.feed(unescape(text))
    # API의 텍스트 표는 행과 셀 구분을 남기고 테두리·패딩만 제거한다.
    text = "".join(parser.parts)
    text = re.sub(r"[│┃]", " | ", text)
    text = re.sub(r"[\u2500-\u257f]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip(" |") for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
