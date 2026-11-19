"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest
from pathlib import Path

from app.core import (
    clean_text,
    detokenize,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    parse_dialogue_line,
    preprocess_corpus,
    tokenize,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
)


# ============================================================================
# Milestone 1: Collect and preprocess a corpus of game dialogues.
# ============================================================================

def test_clean_text():
    raw = "  Guard: ‘Halt!’  “Who goes there?” — State your   business. \n\n"
    cleaned = clean_text(raw)
    assert "‘" not in cleaned and "’" not in cleaned
    assert "“" not in cleaned and "”" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "Guard: 'Halt!' \"Who goes there?\" - State your business."


def test_parse_dialogue_line():
    npc, text = parse_dialogue_line("Guard: Halt! Who goes there?")
    assert npc == "Guard"
    assert text == "Halt! Who goes there?"

    npc2, text2 = parse_dialogue_line("[Shopkeeper] Welcome, traveler.")
    assert npc2 == "Shopkeeper"
    assert text2 == "Welcome, traveler."

    npc3, text3 = parse_dialogue_line("Just wandering without a name.")
    assert npc3 is None
    assert text3 == "Just wandering without a name."


def test_tokenize_preserves_contractions_and_punctuation():
    line = "I don't think you're ready for the dragon's breath, hero!"
    tokens = tokenize(line)
    assert "don't" in tokens
    assert "you're" in tokens
    assert "dragon's" in tokens
    assert "," in tokens
    assert "!" in tokens


def test_detokenize_formats_dialogue():
    tokens = ["<s>", "<s>", "Halt", "!", "Who", "goes", "there", "?", "</s>"]
    text = detokenize(tokens)
    assert text == "Halt! Who goes there?"


def test_preprocess_corpus_padding_and_vocab():
    sample = "Guard: Halt! Who goes there?\nMerchant: Gold speaks louder."
    pre = preprocess_corpus(sample, n=3)
    assert pre.total_sentences == 2
    # Trigram requires 2 START_TOKENs and 1 END_TOKEN
    assert pre.sentences[0][:2] == [START_TOKEN, START_TOKEN]
    assert pre.sentences[0][-1] == END_TOKEN
    assert START_TOKEN in pre.vocabulary
    assert END_TOKEN in pre.vocabulary
    assert UNK_TOKEN in pre.vocabulary
    assert "Guard" in pre.npc_dialogues
    assert "Merchant" in pre.npc_dialogues


def test_milestone_1():
    result = milestone_1()
    assert result["status"] == "success"
    assert result["sentence_count"] > 50
    assert result["vocab_size"] > 100
    assert result["total_tokens"] > 500
    assert "Guard" in result["npc_archetypes"]
    assert "Merchant" in result["npc_archetypes"]
    assert "Mage" in result["npc_archetypes"]


# ============================================================================
# Milestone 2: Build the n-gram language model with smoothing.
# ============================================================================
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# ============================================================================
# Milestone 3: Implement dialogue generation logic using the model.
# ============================================================================
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# ============================================================================
# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
# ============================================================================
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
