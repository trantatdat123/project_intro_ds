"""FlashText, ranh giới từ giữ C++/C#/.NET và tên kỹ năng chuẩn."""

import json
import re
from pathlib import Path

from flashtext import KeywordProcessor

from src.preprocessing.cleaner import normalize_text
from src.utils import ROOT


class SkillExtractor:
    def __init__(self, skill_dict_path=ROOT / "configs/skill_dictionary.json"):
        self.dictionary = json.loads(Path(skill_dict_path).read_text(encoding="utf-8"))
        self.processor = KeywordProcessor(case_sensitive=False)
        # FlashText mặc định chỉ coi chữ ASCII là ký tự trong từ: 'rõ' có thể
        # khớp nhầm R. Bổ sung chữ/số Unicode trước khi đăng ký keyword.
        self.processor.non_word_boundaries.update(chr(i) for i in range(0x10000) if chr(i).isalnum())
        for char in "+#._":
            self.processor.add_non_word_boundary(char)
        aliases_seen = {}
        for canonical, aliases in self.dictionary.items():
            for alias in [canonical, *aliases]:
                key = alias.casefold()
                if key in aliases_seen and aliases_seen[key] != canonical:
                    raise ValueError(f"Alias trùng: {alias}")
                aliases_seen[key] = canonical
                self.processor.add_keyword(alias, canonical)

    def extract_skills(self, text: str) -> list[str]:
        text = normalize_text(text)
        # Dấu chấm kết câu không thuộc tên kỹ năng; giữ node.js/.net và version.
        text = re.sub(r"(?<=[\w+#])\.(?=\s|$)", " ", text)
        return sorted(set(self.processor.extract_keywords(text)))
