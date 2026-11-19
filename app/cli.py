"""Command-line entry point for Game Dialogue Assistant: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import (
    DEFAULT_CORPUS_PATH,
    NGramLanguageModel,
    generate_dialogue_suggestions,
    preprocess_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="game-dialogue-assistant",
        description=(
            "Game Dialogue Assistant: Generates NPC dialogue snippets for game designers "
            "using n-gram language models trained on game scripts to ensure character "
            "consistency and aid rapid prototyping."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # Hello check
    sub.add_parser("hello", help="check that everything is wired up")

    # Generate command
    gen_parser = sub.add_parser("generate", help="generate NPC dialogue snippets from a starting phrase")
    gen_parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="starting phrase for the NPC dialogue",
    )
    gen_parser.add_argument(
        "-p", "--prompt",
        dest="prompt_flag",
        default=None,
        help="starting phrase (alternative to positional argument)",
    )
    gen_parser.add_argument(
        "-l", "--length",
        type=int,
        default=25,
        help="desired length of generated dialogue snippet (default: 25)",
    )
    gen_parser.add_argument(
        "-n", "--suggestions",
        type=int,
        default=3,
        help="number of alternative dialogue suggestions to generate (default: 3)",
    )
    gen_parser.add_argument(
        "-c", "--corpus",
        type=str,
        default=None,
        help="path to custom game dialogue corpus file for training",
    )
    gen_parser.add_argument(
        "--npc",
        type=str,
        default=None,
        help="NPC character archetype to emulate (e.g. Guard, Merchant, Mage, Rogue, Innkeeper)",
    )
    gen_parser.add_argument(
        "-o", "--order",
        type=int,
        default=3,
        help="n-gram model order (default: 3 for trigram)",
    )
    gen_parser.add_argument(
        "-k", "--smoothing",
        type=float,
        default=0.1,
        help="Add-k smoothing parameter (default: 0.1)",
    )
    gen_parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (0.0 for greedy, >0 for diversity; default: 0.7)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible generation",
    )

    # Interactive REPL command
    interactive_parser = sub.add_parser(
        "interactive",
        help="start an interactive session to rapidly prototype NPC dialogues",
    )
    interactive_parser.add_argument(
        "-c", "--corpus",
        type=str,
        default=None,
        help="path to custom game dialogue corpus",
    )
    interactive_parser.add_argument(
        "--npc",
        type=str,
        default=None,
        help="default NPC character archetype filter",
    )
    interactive_parser.add_argument(
        "-l", "--length",
        type=int,
        default=25,
        help="default snippet length",
    )
    interactive_parser.add_argument(
        "-n", "--suggestions",
        type=int,
        default=3,
        help="number of alternative suggestions per prompt",
    )

    # Stats command
    stats_parser = sub.add_parser("stats", help="display corpus statistics and NPC archetypes")
    stats_parser.add_argument(
        "-c", "--corpus",
        type=str,
        default=None,
        help="path to dialogue corpus",
    )

    return parser


def handle_generate(args: argparse.Namespace) -> list[str]:
    """Execute the dialogue generation command."""
    prompt = args.prompt_flag if args.prompt_flag is not None else args.prompt
    if not prompt:
        prompt = input("Enter starting phrase: ").strip()

    npc_filter = args.npc
    corpus_source = args.corpus
    length = max(3, args.length)
    num_suggestions = max(1, args.suggestions)
    temperature = max(0.0, args.temperature)
    n = max(1, args.order)
    k = args.smoothing
    seed = args.seed

    print(f"\nTraining {n}-gram model (k={k}) on corpus...")
    pre = preprocess_corpus(source=corpus_source, n=n, npc_filter=npc_filter)
    if pre.total_sentences == 0:
        print(f"Error: No dialogue sentences found matching filter '{npc_filter}'")
        return []

    model = NGramLanguageModel(n=n, k=k)
    model.train(pre.sentences, vocabulary=pre.vocabulary)

    npc_tag = f" [{npc_filter}]" if npc_filter else ""
    print(f"Generating {num_suggestions} suggestion(s) for starting phrase '{prompt}'{npc_tag}:\n")

    suggestions = model.generate_suggestions(
        prompt=prompt,
        num_suggestions=num_suggestions,
        max_length=length,
        temperature=temperature,
        seed=seed,
    )

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. \"{s}\"")
    print()
    return suggestions


def handle_stats(args: argparse.Namespace) -> None:
    """Print statistics about the game dialogue corpus."""
    corpus_source = args.corpus
    pre = preprocess_corpus(source=corpus_source)
    print("\n--- Game Dialogue Corpus Statistics ---")
    print(f"Total Sentences:    {pre.total_sentences}")
    print(f"Total Tokens:       {pre.total_tokens}")
    print(f"Vocabulary Size:    {pre.vocab_size}")
    print(f"NPC Archetypes ({len(pre.npc_dialogues)}):")
    for npc, dialogues in sorted(pre.npc_dialogues.items()):
        print(f"  - {npc}: {len(dialogues)} dialogues")
    print("---------------------------------------\n")


def handle_interactive(args: argparse.Namespace) -> None:
    """Run an interactive prompt for rapid dialogue prototyping."""
    corpus_source = args.corpus
    npc_filter = args.npc
    length = args.length
    num_suggestions = args.suggestions
    temperature = 0.7
    n = 3
    k = 0.1

    print("=" * 65)
    print("  Game Dialogue Assistant — NPC Prototyping Interface")
    print("=" * 65)
    print("Type a starting phrase to generate dialogue suggestions.")
    print("Commands:")
    print("  /npc <archetype>   - Set NPC filter (e.g. Guard, Merchant, Mage, off)")
    print("  /length <int>      - Change desired snippet length")
    print("  /suggestions <int> - Set number of suggestions")
    print("  /temp <float>      - Set sampling temperature (e.g. 0.3, 0.7, 1.0)")
    print("  /stats             - Display corpus statistics")
    print("  quit / exit        - Exit the tool\n")

    pre = preprocess_corpus(source=corpus_source, n=n, npc_filter=npc_filter)
    model = NGramLanguageModel(n=n, k=k)
    model.train(pre.sentences, vocabulary=pre.vocabulary)

    while True:
        try:
            npc_indicator = f"[{npc_filter}] " if npc_filter else ""
            user_input = input(f"{npc_indicator}Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Happy game designing!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            print("Exiting. Happy game designing!")
            break

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/npc":
                if not arg or arg.lower() in ("none", "off", "all"):
                    npc_filter = None
                    print("NPC filter cleared. Using all character dialogues.")
                else:
                    npc_filter = arg
                    print(f"NPC filter set to: {npc_filter}")
                # Retrain model with new filter
                pre = preprocess_corpus(source=corpus_source, n=n, npc_filter=npc_filter)
                model = NGramLanguageModel(n=n, k=k)
                model.train(pre.sentences, vocabulary=pre.vocabulary)
            elif cmd == "/length":
                try:
                    length = max(3, int(arg))
                    print(f"Dialogue snippet length set to: {length}")
                except ValueError:
                    print("Usage: /length <number>")
            elif cmd == "/suggestions":
                try:
                    num_suggestions = max(1, int(arg))
                    print(f"Suggestions count set to: {num_suggestions}")
                except ValueError:
                    print("Usage: /suggestions <number>")
            elif cmd == "/temp":
                try:
                    temperature = max(0.0, float(arg))
                    print(f"Sampling temperature set to: {temperature}")
                except ValueError:
                    print("Usage: /temp <float>")
            elif cmd == "/stats":
                handle_stats(args)
            elif cmd == "/help":
                print("Available commands: /npc <name>, /length <int>, /suggestions <int>, /temp <float>, /stats, quit")
            else:
                print(f"Unknown command: {cmd}. Type /help for options.")
            continue

        # Generate dialogue suggestions
        suggestions = model.generate_suggestions(
            prompt=user_input,
            num_suggestions=num_suggestions,
            max_length=length,
            temperature=temperature,
        )

        print(f"\nGenerated {len(suggestions)} suggestion(s):")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. \"{s}\"")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    
    # If no arguments passed, launch interactive mode
    if argv is not None and len(argv) == 0:
        handle_interactive(parser.parse_args(["interactive"]))
        return
    elif argv is None and len(sys.argv) == 1:
        handle_interactive(parser.parse_args(["interactive"]))
        return

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Game Dialogue Assistant is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "stats":
        handle_stats(args)
    elif args.command == "interactive":
        handle_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
