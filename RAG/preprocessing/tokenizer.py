"""문서와 질의에 동일하게 적용하는 한국어 토큰화."""

from importlib.metadata import version
import re
import unicodedata


class KoreanTokenizer:
    def __init__(self):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi(num_workers=1, model_type="cong")

    def describe(self):
        return {
            "strategy": "kiwi_lexical_v1",
            "normalization": "NFKC_lowercase",
            "model_type": "cong",
            "removed_pos_prefixes": ["J", "E"],
            "removed_tags": ["SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW"],
            "keep_negation_and_numbers": True,
            "preserve_article_references": True,
            "kiwipiepy_version": version("kiwipiepy"),
            "kiwipiepy_model_version": version("kiwipiepy_model"),
        }

    def tokenize(self, text):
        if not isinstance(text, str):
            raise ValueError("토큰화할 입력은 문자열이어야 합니다.")
        text = unicodedata.normalize("NFKC", text).lower()
        tokens = []
        punctuation = {"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW"}
        for token in self.kiwi.tokenize(text):
            if token.tag.startswith(("J", "E")) or token.tag in punctuation:
                continue
            form = token.form.strip()
            if form and any(ch.isalnum() for ch in form):
                tokens.append(form)
        for match in re.finditer(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", text):
            number, branch = match.groups()
            tokens.append(f"article:{int(number)}:{int(branch or 0)}")
        return tokens
