"""Core logic for Game Dialogue Assistant:
Corpus preprocessing, n-gram language modeling with smoothing, dialogue generation, and CLI helpers.
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Default location of the game dialogues corpus
DEFAULT_CORPUS_PATH = Path(__file__).parent / "data" / "game_dialogues.txt"

# Special tokens
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"


# ============================================================================
# Milestone 1: Collect and Preprocess a Corpus of Game Dialogues
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize raw dialogue text.
    
    Normalizes curly quotes, removes excessive whitespace, and strips outer spaces.
    """
    if not text:
        return ""
    # Normalize curly apostrophes and quotation marks
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace tabs and multiple whitespace characters with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading and trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def parse_dialogue_line(line: str) -> tuple[str | None, str]:
    """Parse an NPC dialogue line, separating NPC character name from the dialogue.
    
    Examples:
        "Guard: Halt! Who goes there?" -> ("Guard", "Halt! Who goes there?")
        "[Merchant] Welcome traveler!"  -> ("Merchant", "Welcome traveler!")
        "Greetings, hero."              -> (None, "Greetings, hero.")
    """
    line = line.strip()
    # Match "NPCName: Dialogue"
    match = re.match(r"^([A-Za-z0-9_\s]{1,30}):\s*(.+)$", line)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Match "[NPCName] Dialogue"
    match_bracket = re.match(r"^\[([A-Za-z0-9_\s]{1,30})\]\s*(.+)$", line)
    if match_bracket:
        return match_bracket.group(1).strip(), match_bracket.group(2).strip()
    return None, line


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Tokenize a dialogue string into words, contractions, and punctuation.
    
    Contractions like "don't", "it's", "you'll" and possessives like "dragon's"
    are preserved as single tokens. Punctuation marks are separated.
    Special tokens like <s>, </s>, <UNK> are preserved.
    """
    if not text:
        return []
    if lowercase:
        text = text.lower()
    
    # Regex pattern:
    # 1. Special tokens like <s>, </s>, <UNK>
    # 2. Word tokens with internal apostrophes (contractions / possessives)
    # 3. Numbers / digits
    # 4. Standard punctuation marks
    pattern = re.compile(
        r"<[A-Za-z0-9_/-]+>|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[.,!?;:\"'\(\)\-]"
    )
    return pattern.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Reconstruct natural, human-readable text from a sequence of tokens.
    
    Removes boundary tokens, attaches punctuation without unwanted preceding
    spaces, and properly capitalizes sentences.
    """
    if not tokens:
        return ""
    
    # Filter out start/end boundary tokens
    clean_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
    if not clean_tokens:
        return ""
    
    # Reassemble tokens with smart spacing
    no_space_before = {".", ",", "!", "?", ";", ":", "'", ")", "]", "}", "”", '"'}
    no_space_after = {"(", "[", "{", "“", '"'}
    
    result: list[str] = []
    prev_token = ""
    in_quote = False
    
    for token in clean_tokens:
        if token == '"':
            # Toggle quotation mark spacing
            if in_quote:
                result.append(token)
                in_quote = False
            else:
                if result and result[-1] != " ":
                    result.append(" ")
                result.append(token)
                in_quote = True
            prev_token = token
            continue
            
        if not result:
            result.append(token)
        elif token in no_space_before:
            result.append(token)
        elif prev_token in no_space_after:
            result.append(token)
        elif prev_token == "'":
            result.append(token)
        else:
            result.append(" " + token)
            
        prev_token = token
    
    text = "".join(result).strip()
    
    # Post-process: capitalize the first character and characters following . ! ?
    def _capitalize_match(match: re.Match) -> str:
        prefix = match.group(1)
        char = match.group(2)
        return prefix + char.upper()

    # Capitalize start of text
    if text:
        text = text[0].upper() + text[1:]
    # Capitalize after sentence terminators
    text = re.sub(r"([.!?]\s+)([a-z])", _capitalize_match, text)
    return text


@dataclass
class PreprocessedCorpus:
    """Container for preprocessed dialogue corpus data."""
    sentences: list[list[str]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    npc_dialogues: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    vocabulary: set[str] = field(default_factory=set)
    vocab_counts: Counter[str] = field(default_factory=Counter)
    n: int = 3

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def total_tokens(self) -> int:
        return sum(self.vocab_counts.values())

    @property
    def total_sentences(self) -> int:
        return len(self.sentences)


def load_corpus_text(path_or_text: str | Path | None = None) -> str:
    """Load dialogue text from a file path or return input text directly."""
    if path_or_text is None:
        path = DEFAULT_CORPUS_PATH
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Default corpus not found at {path}")
    
    # Check if path_or_text is an existing file
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    
    # Otherwise treat as raw text content
    return str(path_or_text)


def preprocess_corpus(
    source: str | Path | None = None,
    n: int = 3,
    npc_filter: str | None = None,
    lowercase: bool = False,
) -> PreprocessedCorpus:
    """Load, clean, tokenize, and prepare a corpus of game dialogues.
    
    Pads each sentence with (n - 1) START_TOKENs and one END_TOKEN.
    Supports filtering by NPC character name for character-consistent dialogue models.
    """
    raw_text = load_corpus_text(source)
    cleaned = clean_text(raw_text)
    lines = cleaned.splitlines()
    
    sentences: list[list[str]] = []
    npc_dialogues: dict[str, list[str]] = defaultdict(list)
    vocab_counts: Counter[str] = Counter()
    
    pad_start = [START_TOKEN] * (n - 1)
    
    for line in lines:
        if not line:
            continue
        npc, dialogue = parse_dialogue_line(line)
        if npc:
            npc_dialogues[npc].append(dialogue)
        
        # If an NPC filter is provided, skip dialogues from non-matching NPCs
        if npc_filter:
            if not npc or npc_filter.lower() not in npc.lower():
                continue
                
        tokens = tokenize(dialogue, lowercase=lowercase)
        if not tokens:
            continue
            
        padded_sentence = pad_start + tokens + [END_TOKEN]
        sentences.append(padded_sentence)
        for token in padded_sentence:
            vocab_counts[token] += 1
            
    vocab = set(vocab_counts.keys())
    # Ensure special tokens are represented in vocabulary
    vocab.add(START_TOKEN)
    vocab.add(END_TOKEN)
    vocab.add(UNK_TOKEN)
    
    return PreprocessedCorpus(
        sentences=sentences,
        raw_lines=lines,
        npc_dialogues=npc_dialogues,
        vocabulary=vocab,
        vocab_counts=vocab_counts,
        n=n,
    )


def milestone_1(source: str | Path | None = None, npc_filter: str | None = None) -> dict[str, Any]:
    """Milestone 1: Collect and preprocess a corpus of game dialogues.
    
    Returns a dictionary summarizing text cleaning, tokenization, vocabulary,
    and dialogue breakdown.
    """
    corpus = preprocess_corpus(source=source, n=3, npc_filter=npc_filter)
    return {
        "status": "success",
        "sentence_count": corpus.total_sentences,
        "vocab_size": corpus.vocab_size,
        "total_tokens": corpus.total_tokens,
        "sample_tokens": corpus.sentences[0] if corpus.sentences else [],
        "npc_archetypes": list(corpus.npc_dialogues.keys()),
        "corpus_object": corpus,
    }


# ============================================================================
# Milestone 2: Build the N-gram Language Model with Smoothing
# ============================================================================

def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# ============================================================================
# Milestone 3: Implement Dialogue Generation Logic Using the Model
# ============================================================================

def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# ============================================================================
# Milestone 4: Design a CLI Where Users Can Input a Starting Phrase
# ============================================================================

def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
