from typing import Optional, List

from dataclasses import dataclass, field


@dataclass
class Token:
    token: Optional[str] = None
    id: Optional[int] = None
    lemma: Optional[str] = None
    upostag: Optional[str] = None
    xpostag: Optional[str] = None
    head: Optional[int] = None
    deprel: Optional[str] = None
    feats: Optional[str] = None

    @classmethod
    def from_json(cls, json):
        return cls(**json)


@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)

    @classmethod
    def from_json(cls, json):
        return cls(tokens=[Token.from_json(t) for t in json["tree"]],
                   embedding=json.get("sentence_embedding", []))
