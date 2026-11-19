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
from typing import Any, Iterable, Sequence


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
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", " - ")
    text = re.sub(r"[ \t]+", " ", text)
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
    match = re.match(r"^([A-Za-z0-9_\s]{1,30}):\s*(.+)$", line)
    if match:
        return match.group(1).strip(), match.group(2).strip()
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
    
    clean_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
    if not clean_tokens:
        return ""
    
    no_space_before = {".", ",", "!", "?", ";", ":", "'", ")", "]", "}", "”", '"'}
    no_space_after = {"(", "[", "{", "“", '"'}
    
    result: list[str] = []
    prev_token = ""
    in_quote = False
    
    for token in clean_tokens:
        if token == '"':
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
    
    def _capitalize_match(match: re.Match) -> str:
        prefix = match.group(1)
        char = match.group(2)
        return prefix + char.upper()

    if text:
        text = text[0].upper() + text[1:]
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
    
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    
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

class NGramLanguageModel:
    """N-gram language model with Add-k (Lidstone / Laplace) smoothing and backoff.
    
    Attributes:
        n: Order of the n-gram model (e.g. 2 for bigram, 3 for trigram).
        k: Add-k smoothing parameter (k > 0).
        vocabulary: Set of known tokens in vocabulary.
        ngram_counts: Mapping from context tuple to Counter of next word occurrences.
        context_counts: Mapping from context tuple to total count of occurrences.
    """

    def __init__(self, n: int = 3, k: float = 0.1, unk_token: str = UNK_TOKEN):
        if n < 1:
            raise ValueError(f"N-gram order n must be at least 1, got {n}")
        if k <= 0:
            raise ValueError(f"Smoothing parameter k must be positive, got {k}")
        self.n = n
        self.k = k
        self.unk_token = unk_token
        self.vocabulary: set[str] = {START_TOKEN, END_TOKEN, unk_token}
        self.ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.backoff_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self.backoff_context_counts: dict[int, Counter[tuple[str, ...]]] = defaultdict(Counter)
        self.unigram_counts: Counter[str] = Counter()
        self.total_tokens: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def train(self, sentences: Iterable[Sequence[str]], vocabulary: Iterable[str] | None = None) -> None:
        """Train the n-gram model on tokenized, padded sentences."""
        if vocabulary:
            self.vocabulary.update(vocabulary)
        self.vocabulary.add(START_TOKEN)
        self.vocabulary.add(END_TOKEN)
        self.vocabulary.add(self.unk_token)

        for sentence in sentences:
            for token in sentence:
                self.vocabulary.add(token)
                self.unigram_counts[token] += 1
                self.total_tokens += 1

            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i : i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

            for order in range(1, self.n):
                for i in range(len(sentence) - order):
                    sub_ngram = tuple(sentence[i : i + order + 1])
                    sub_ctx = sub_ngram[:-1]
                    sub_word = sub_ngram[-1]
                    self.backoff_counts[order][sub_ctx][sub_word] += 1
                    self.backoff_context_counts[order][sub_ctx] += 1

    def _normalize_token(self, token: str) -> str:
        """Map unknown tokens to UNK_TOKEN."""
        return token if token in self.vocabulary else self.unk_token

    def _normalize_context(self, context: Sequence[str], length: int | None = None) -> tuple[str, ...]:
        """Normalize and pad or trim context to the requested length (default n - 1)."""
        target_len = self.n - 1 if length is None else length
        ctx_tokens = [self._normalize_token(t) for t in context]
        if len(ctx_tokens) < target_len:
            ctx_tokens = [START_TOKEN] * (target_len - len(ctx_tokens)) + ctx_tokens
        elif len(ctx_tokens) > target_len:
            ctx_tokens = ctx_tokens[-target_len:]
        return tuple(ctx_tokens)

    def probability(self, context: Sequence[str], word: str, use_backoff: bool = True) -> float:
        """Calculate the smoothed probability P(word | context) with Add-k smoothing."""
        norm_word = self._normalize_token(word)
        norm_ctx = self._normalize_context(context)
        v_size = self.vocab_size

        ctx_count = self.context_counts[norm_ctx]
        if ctx_count > 0:
            word_count = self.ngram_counts[norm_ctx][norm_word]
            return (word_count + self.k) / (ctx_count + self.k * v_size)

        if not use_backoff or self.n <= 1:
            return self.k / (ctx_count + self.k * v_size)

        for order in range(self.n - 1, 0, -1):
            sub_ctx = norm_ctx[-(order - 1):] if order > 1 else ()
            sub_ctx_count = self.backoff_context_counts[order][sub_ctx]
            if sub_ctx_count > 0:
                sub_word_count = self.backoff_counts[order][sub_ctx][norm_word]
                return (sub_word_count + self.k) / (sub_ctx_count + self.k * v_size)

        unigram_count = self.unigram_counts[norm_word]
        return (unigram_count + self.k) / (self.total_tokens + self.k * v_size)

    def get_distribution(self, context: Sequence[str], use_backoff: bool = True) -> dict[str, float]:
        """Compute the next-word probability distribution for a given context."""
        norm_ctx = self._normalize_context(context)
        v_size = self.vocab_size
        ctx_count = self.context_counts[norm_ctx]

        if ctx_count > 0:
            denom = ctx_count + self.k * v_size
            counts = self.ngram_counts[norm_ctx]
            return {word: (counts[word] + self.k) / denom for word in self.vocabulary}

        if not use_backoff or self.n <= 1:
            prob = 1.0 / v_size
            return {w: prob for w in self.vocabulary}

        for order in range(self.n - 1, 0, -1):
            sub_ctx = norm_ctx[-(order - 1):] if order > 1 else ()
            sub_ctx_count = self.backoff_context_counts[order][sub_ctx]
            if sub_ctx_count > 0:
                denom = sub_ctx_count + self.k * v_size
                counts = self.backoff_counts[order][sub_ctx]
                return {w: (counts[w] + self.k) / denom for w in self.vocabulary}

        denom = self.total_tokens + self.k * v_size
        return {w: (self.unigram_counts[w] + self.k) / denom for w in self.vocabulary}

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Calculate model perplexity on a test set of tokenized sentences."""
        log_prob_sum = 0.0
        token_count = 0

        for sentence in sentences:
            for i in range(self.n - 1, len(sentence)):
                context = tuple(sentence[i - self.n + 1 : i])
                word = sentence[i]
                prob = self.probability(context, word)
                log_prob_sum += math.log(max(prob, 1e-12))
                token_count += 1

        if token_count == 0:
            return float("inf")
        return math.exp(-log_prob_sum / token_count)

    def _get_generation_candidates(self, context: tuple[str, ...]) -> dict[str, float]:
        """Get candidate words and their Add-k smoothed probabilities for text generation.
        
        Prioritizes words observed in the n-gram context, backing off gracefully
        if the context is unseen.
        """
        # Primary context check
        if context in self.ngram_counts and self.ngram_counts[context]:
            counts = self.ngram_counts[context]
            denom = self.context_counts[context] + self.k * len(counts)
            return {
                w: (c + self.k) / denom
                for w, c in counts.items()
                if w not in (START_TOKEN, self.unk_token)
            }

        # Backoff levels (from n-1 down to 1)
        for order in range(self.n - 1, 0, -1):
            sub_ctx = context[-(order - 1):] if order > 1 else ()
            if sub_ctx in self.backoff_counts[order] and self.backoff_counts[order][sub_ctx]:
                counts = self.backoff_counts[order][sub_ctx]
                denom = self.backoff_context_counts[order][sub_ctx] + self.k * len(counts)
                return {
                    w: (c + self.k) / denom
                    for w, c in counts.items()
                    if w not in (START_TOKEN, self.unk_token)
                }

        # Fallback to unigram counts
        denom = self.total_tokens + self.k * len(self.unigram_counts)
        return {
            w: (c + self.k) / denom
            for w, c in self.unigram_counts.items()
            if w not in (START_TOKEN, self.unk_token)
        }

    def generate(
        self,
        prompt: str = "",
        max_length: int = 25,
        min_length: int = 3,
        temperature: float = 0.7,
        top_k: int = 20,
        seed: int | None = None,
        stop_on_punct: bool = False,
    ) -> str:
        """Generate a dialogue snippet starting from prompt.
        
        Args:
            prompt: Initial starting phrase.
            max_length: Desired maximum length in tokens.
            min_length: Minimum tokens before END_TOKEN can terminate.
            temperature: Sampling temperature (0.0 for greedy argmax, >0 for diversity).
            top_k: Restrict sampling to top K candidate tokens.
            seed: Optional random seed for reproducible output.
            stop_on_punct: Stop after terminal punctuation once min_length is reached.
        """
        rng = random.Random(seed) if seed is not None else random
        
        prompt_tokens = tokenize(prompt)
        tokens = list(prompt_tokens)
        
        pad_needed = max(0, (self.n - 1) - len(tokens))
        context_list = [START_TOKEN] * pad_needed + tokens
        context = tuple(context_list[-(self.n - 1):]) if self.n > 1 else ()

        terminal_punct = {".", "!", "?"}
        
        for _ in range(max_length - len(prompt_tokens)):
            candidates = self._get_generation_candidates(context)
            
            # Suppress END_TOKEN before min_length is reached
            if len(tokens) < min_length and END_TOKEN in candidates and len(candidates) > 1:
                candidates = {w: p for w, p in candidates.items() if w != END_TOKEN}
                
            if not candidates:
                break
                
            sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
            if top_k > 0 and len(sorted_candidates) > top_k:
                sorted_candidates = sorted_candidates[:top_k]
                
            words, raw_probs = zip(*sorted_candidates)
            
            if temperature <= 0.001:
                next_word = words[0]
            else:
                scaled_probs = [p ** (1.0 / temperature) for p in raw_probs]
                sum_scaled = sum(scaled_probs)
                norm_probs = [p / sum_scaled for p in scaled_probs]
                next_word = rng.choices(words, weights=norm_probs, k=1)[0]
                
            if next_word == END_TOKEN:
                break
                
            tokens.append(next_word)
            
            if stop_on_punct and next_word in terminal_punct and len(tokens) >= min_length:
                break
                
            if self.n > 1:
                context = tuple((list(context) + [next_word])[-(self.n - 1):])

        return detokenize(tokens)

    def generate_suggestions(
        self,
        prompt: str = "",
        num_suggestions: int = 3,
        max_length: int = 25,
        min_length: int = 3,
        temperature: float = 0.7,
        top_k: int = 20,
        seed: int | None = None,
    ) -> list[str]:
        """Generate multiple distinct dialogue alternatives for a starting phrase."""
        suggestions: list[str] = []
        seen: set[str] = set()
        base_seed = seed if seed is not None else random.randint(0, 1000000)
        
        attempts = 0
        max_attempts = num_suggestions * 5
        
        while len(suggestions) < num_suggestions and attempts < max_attempts:
            current_seed = base_seed + attempts
            dialogue = self.generate(
                prompt=prompt,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                seed=current_seed,
            )
            if dialogue and dialogue not in seen:
                seen.add(dialogue)
                suggestions.append(dialogue)
            attempts += 1
            
        return suggestions


def milestone_2(
    corpus: PreprocessedCorpus | None = None,
    n: int = 3,
    k: float = 0.1,
) -> dict[str, Any]:
    """Milestone 2: Build the n-gram language model with smoothing.
    
    Constructs an n-gram model with Add-k smoothing, trains it on the corpus,
    and returns the model along with training metrics.
    """
    if corpus is None:
        corpus = preprocess_corpus(n=n)
    
    model = NGramLanguageModel(n=n, k=k)
    model.train(corpus.sentences, vocabulary=corpus.vocabulary)
    
    sample_context = (START_TOKEN,) * (n - 1)
    dist = model.get_distribution(sample_context)
    dist_sum = sum(dist.values())
    ppl = model.perplexity(corpus.sentences[:20])
    
    return {
        "status": "success",
        "model": model,
        "n": n,
        "k": k,
        "vocab_size": model.vocab_size,
        "distinct_contexts": len(model.context_counts),
        "total_tokens_trained": model.total_tokens,
        "sample_distribution_sum": dist_sum,
        "perplexity_sample": ppl,
    }


# ============================================================================
# Milestone 3: Implement Dialogue Generation Logic Using the Model
# ============================================================================

def generate_dialogue(
    prompt: str = "Greetings traveler",
    model: NGramLanguageModel | None = None,
    corpus_source: str | Path | None = None,
    npc_filter: str | None = None,
    n: int = 3,
    k: float = 0.1,
    max_length: int = 25,
    min_length: int = 3,
    temperature: float = 0.7,
    top_k: int = 20,
    seed: int | None = None,
) -> str:
    """Generate a dialogue snippet using an n-gram model trained on game scripts.
    
    Supports custom corpus, NPC filtering for character consistency, length limit,
    and temperature-controlled generation.
    """
    if model is None:
        corpus = preprocess_corpus(source=corpus_source, n=n, npc_filter=npc_filter)
        model = NGramLanguageModel(n=n, k=k)
        model.train(corpus.sentences, vocabulary=corpus.vocabulary)

    return model.generate(
        prompt=prompt,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


def generate_dialogue_suggestions(
    prompt: str = "Greetings traveler",
    num_suggestions: int = 3,
    model: NGramLanguageModel | None = None,
    corpus_source: str | Path | None = None,
    npc_filter: str | None = None,
    n: int = 3,
    k: float = 0.1,
    max_length: int = 25,
    min_length: int = 3,
    temperature: float = 0.7,
    top_k: int = 20,
    seed: int | None = None,
) -> list[str]:
    """Generate multiple alternative dialogue suggestions for a given starting phrase."""
    if model is None:
        corpus = preprocess_corpus(source=corpus_source, n=n, npc_filter=npc_filter)
        model = NGramLanguageModel(n=n, k=k)
        model.train(corpus.sentences, vocabulary=corpus.vocabulary)

    return model.generate_suggestions(
        prompt=prompt,
        num_suggestions=num_suggestions,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


def milestone_3(
    prompt: str = "Greetings traveler",
    max_length: int = 25,
    num_suggestions: int = 3,
    temperature: float = 0.7,
    seed: int | None = 42,
    npc_filter: str | None = None,
) -> dict[str, Any]:
    """Milestone 3: Implement dialogue generation logic using the model.
    
    Generates a coherent dialogue snippet and alternative suggestions given
    a starting phrase.
    """
    corpus = preprocess_corpus(n=3, npc_filter=npc_filter)
    model = NGramLanguageModel(n=3, k=0.1)
    model.train(corpus.sentences, vocabulary=corpus.vocabulary)
    
    primary_dialogue = model.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        seed=seed,
    )
    
    suggestions = model.generate_suggestions(
        prompt=prompt,
        num_suggestions=num_suggestions,
        max_length=max_length,
        temperature=temperature,
        seed=seed,
    )
    
    return {
        "status": "success",
        "prompt": prompt,
        "dialogue": primary_dialogue,
        "suggestions": suggestions,
        "length": max_length,
        "num_suggestions": len(suggestions),
    }


# ============================================================================
# Milestone 4: Design a CLI Where Users Can Input a Starting Phrase
# ============================================================================

def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
