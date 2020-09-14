import collections
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple

import conllu
from dataclasses_json import dataclass_json
from overrides import overrides


@dataclass_json
@dataclass
class Token:
    id: Optional[Union[int, Tuple]] = None
    token: Optional[str] = None
    lemma: Optional[str] = None
    upostag: Optional[str] = None
    xpostag: Optional[str] = None
    feats: Optional[str] = None
    head: Optional[int] = None
    deprel: Optional[str] = None
    deps: Optional[str] = None
    misc: Optional[str] = None
    semrel: Optional[str] = None


@dataclass_json
@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    sentence_embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=collections.OrderedDict)


class _TokenList(conllu.TokenList):

    @overrides
    def __repr__(self):
        return 'TokenList<' + ', '.join(token['token'] for token in self) + '>'


def sentence2conllu(sentence: Sentence, keep_semrel: bool = True) -> conllu.TokenList:
    tokens = []
    for token in sentence.tokens:
        token_dict = collections.OrderedDict(token.to_dict())
        # Remove semrel to have default conllu format.
        if not keep_semrel:
            del token_dict["semrel"]
        tokens.append(token_dict)
    # Range tokens must be tuple not list, this is conllu library requirement
    for t in tokens:
        if type(t["id"]) == list:
            t["id"] = tuple(t["id"])
    return _TokenList(tokens=tokens,
                      metadata=sentence.metadata)


def tokens2conllu(tokens: List[str]) -> conllu.TokenList:
    return _TokenList(
        [collections.OrderedDict({"id": idx, "token": token}) for
         idx, token
         in enumerate(tokens, start=1)],
        metadata=collections.OrderedDict()
    )


def conllu2sentence(conllu_sentence: conllu.TokenList,
                    sentence_embedding: List[float]) -> Sentence:
    return Sentence(
        tokens=[Token.from_dict(t) for t in conllu_sentence.tokens],
        sentence_embedding=sentence_embedding,
        metadata=conllu_sentence.metadata
    )
