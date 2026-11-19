# Game Dialogue Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2nmhs2k8r5/badge.svg)](https://ask.thili.ai/b/2nmhs2k8r5)

This Game Dialogue Assistant is a tool for game designers in the gaming domain. It helps generate dialogue for Non-Player Characters (NPCs) by leveraging a next-word prediction model trained on existing game scripts. This ensures character consistency and speeds up the dialogue creation process.

## What you are building

**Goal.** You will build a command-line tool that helps game designers generate dialogue snippets for NPCs, ensuring character consistency and aiding rapid prototyping.

**Done when.** You will have a functional command-line interface where a user can input a starting phrase, and the tool will generate a coherent dialogue snippet based on a corpus of game scripts. The generated dialogue should reflect patterns learned from the training data.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Focus on text cleaning, tokenization, and preparing your text data.
- [x] **2. Build the n-gram language model with smoothing.**  
      Hint: Implement n-gram counting and apply a smoothing technique like Add-k smoothing.
- [x] **3. Implement dialogue generation logic using the model.**  
      Hint: Use your n-gram model to iteratively predict the next word to form a sequence.
- [x] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Create an interactive command-line interface to take user input and display generated text.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- [x] Allow users to specify the desired length of the generated dialogue snippet (`-l` / `--length`).
- [x] Offer multiple alternative dialogue suggestions for a given starting phrase (`-n` / `--suggestions`).
- [x] Enable users to load their own custom game dialogue corpus for model training (`-c` / `--corpus`).
- [x] Character consistency conditioning via NPC archetype filtering (`--npc Guard|Merchant|Mage|...`).

## CLI Usage

### Generate Dialogue Snippets
```bash
# Generate 3 alternative suggestions from a starting phrase
python -m app.cli generate "Greetings traveler" --length 20 --suggestions 3

# Target specific NPC archetype for character consistency
python -m app.cli generate "Halt" --npc Guard --length 15 --suggestions 2
python -m app.cli generate "The arcane" --npc Mage --suggestions 2
python -m app.cli generate "A dull blade" --npc Blacksmith --suggestions 2

# Load a custom game dialogue script corpus
python -m app.cli generate "Welcome" --corpus path/to/dialogues.txt --length 20
```

### Interactive Prototyping REPL
```bash
python -m app.cli interactive
```
In interactive mode, designers can test prompts and configure options on the fly:
- Enter any starting phrase to generate suggestions
- `/npc <name>`: Switch persona (e.g. `/npc Guard`, `/npc Mage`, `/npc off`)
- `/length <int>`: Set snippet length
- `/suggestions <int>`: Set number of alternatives
- `/temp <float>`: Adjust sampling temperature
- `/stats`: View corpus and archetype breakdown
- `quit` / `exit`: Exit the tool

### Corpus Statistics
```bash
python -m app.cli stats
```
