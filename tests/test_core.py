"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest
import math
from pathlib import Path

from app.cli import main
from app.core import (
    clean_text,
    detokenize,
    generate_dialogue,
    generate_dialogue_suggestions,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    parse_dialogue_line,
    preprocess_corpus,
    tokenize,
    NGramLanguageModel,
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

def test_ngram_counting():
    sentences = [
        ["<s>", "<s>", "Halt", "!", "</s>"],
        ["<s>", "<s>", "Halt", ",", "stranger", "</s>"],
    ]
    model = NGramLanguageModel(n=3, k=1.0)
    model.train(sentences)
    
    assert model.context_counts[("<s>", "<s>")] == 2
    assert model.ngram_counts[("<s>", "<s>")]["Halt"] == 2
    assert model.context_counts[("<s>", "Halt")] == 2
    assert model.ngram_counts[("<s>", "Halt")]["!"] == 1
    assert model.ngram_counts[("<s>", "Halt")][","] == 1


def test_add_k_smoothing_probabilities_sum_to_one():
    sentences = [
        ["<s>", "<s>", "Halt", "!", "</s>"],
        ["<s>", "<s>", "Who", "goes", "there", "?", "</s>"],
    ]
    model = NGramLanguageModel(n=3, k=0.5)
    model.train(sentences)
    
    context = ("<s>", "<s>")
    dist = model.get_distribution(context)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-5)
    
    unseen_ctx = ("dragons", "flying")
    dist_unseen = model.get_distribution(unseen_ctx)
    assert math.isclose(sum(dist_unseen.values()), 1.0, rel_tol=1e-5)


def test_add_k_smoothing_assigns_nonzero_probability():
    sentences = [
        ["<s>", "<s>", "Stay", "a", "while", "</s>"],
    ]
    model = NGramLanguageModel(n=3, k=0.1)
    model.train(sentences)
    
    prob = model.probability(("<s>", "<s>"), "dragon")
    assert prob > 0.0


def test_perplexity_calculation():
    sentences = [
        ["<s>", "<s>", "Halt", "!", "Who", "goes", "there", "?", "</s>"],
        ["<s>", "<s>", "Welcome", "traveler", "to", "my", "shop", "</s>"],
    ]
    model = NGramLanguageModel(n=3, k=0.1)
    model.train(sentences)
    
    ppl = model.perplexity(sentences)
    assert ppl > 0
    assert not math.isinf(ppl)
    assert not math.isnan(ppl)


def test_milestone_2():
    result = milestone_2()
    assert result["status"] == "success"
    assert result["n"] == 3
    assert result["k"] == 0.1
    assert result["vocab_size"] > 100
    assert result["distinct_contexts"] > 50
    assert math.isclose(result["sample_distribution_sum"], 1.0, rel_tol=1e-5)
    assert result["perplexity_sample"] > 0


# ============================================================================
# Milestone 3: Implement dialogue generation logic using the model.
# ============================================================================

def test_generate_dialogue_starts_with_prompt():
    prompt = "Welcome to my"
    dialogue = generate_dialogue(prompt=prompt, seed=42)
    assert dialogue.lower().startswith("welcome to my")
    assert len(dialogue.split()) > len(prompt.split())


def test_generate_dialogue_respects_max_length():
    prompt = "The forge"
    max_len = 10
    dialogue = generate_dialogue(prompt=prompt, max_length=max_len, seed=123)
    tokens = tokenize(dialogue)
    assert len(tokens) <= max_len + 1


def test_generate_suggestions_returns_multiple_alternatives():
    prompt = "Keep your"
    suggestions = generate_dialogue_suggestions(prompt=prompt, num_suggestions=3, seed=10)
    assert len(suggestions) >= 2
    for s in suggestions:
        assert s.lower().startswith("keep your")


def test_dialogue_generation_npc_consistency():
    guard_dialogue = generate_dialogue(
        prompt="Halt",
        npc_filter="Guard",
        seed=1,
    )
    assert "halt" in guard_dialogue.lower()


def test_dialogue_generation_custom_corpus(tmp_path):
    custom_file = tmp_path / "custom.txt"
    custom_file.write_text(
        "Alien: Greetings earthling! We come in peace from Mars.\n"
        "Alien: Beware the cosmic ray beam from our mothership.\n"
    )
    dialogue = generate_dialogue(
        prompt="Greetings earthling",
        corpus_source=str(custom_file),
        seed=42,
    )
    assert "greetings earthling" in dialogue.lower()
    assert any(w in dialogue.lower() for w in ["peace", "mars", "we", "come"])


def test_milestone_3():
    result = milestone_3(prompt="Welcome", max_length=20, num_suggestions=3, seed=99)
    assert result["status"] == "success"
    assert result["prompt"] == "Welcome"
    assert result["dialogue"].lower().startswith("welcome")
    assert len(result["suggestions"]) >= 1
    assert result["num_suggestions"] >= 1


# ============================================================================
# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
# ============================================================================

def test_cli_generate_command(capsys):
    main(["generate", "Welcome to", "--length", "20", "--suggestions", "2", "--seed", "42"])
    out = capsys.readouterr().out
    assert "suggestion(s) for starting phrase 'Welcome to'" in out
    assert '1. "' in out
    assert '2. "' in out


def test_cli_npc_filter_and_length(capsys):
    main(["generate", "Halt", "--npc", "Guard", "--length", "15", "--suggestions", "1", "--seed", "7"])
    out = capsys.readouterr().out
    assert "[Guard]" in out
    assert "Halt" in out


def test_cli_custom_corpus(tmp_path, capsys):
    custom_file = tmp_path / "custom_game.txt"
    custom_file.write_text("Pirate: Ahoy matey! Sail the high seas for hidden treasure.\n")
    main(["generate", "Ahoy", "--corpus", str(custom_file), "--suggestions", "1", "--seed", "1"])
    out = capsys.readouterr().out
    assert "Ahoy" in out
    assert any(w in out.lower() for w in ["matey", "sail", "treasure", "seas"])


def test_cli_stats_command(capsys):
    main(["stats"])
    out = capsys.readouterr().out
    assert "Game Dialogue Corpus Statistics" in out
    assert "Total Sentences:" in out
    assert "Vocabulary Size:" in out
    assert "Guard:" in out


def test_milestone_4(capsys):
    result = milestone_4(prompt="Halt", length=20, suggestions=2, npc="Guard")
    assert result["status"] == "success"
    assert result["prompt"] == "Halt"
    assert len(result["suggestions"]) >= 1
