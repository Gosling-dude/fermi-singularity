"""Terminal interface — the guaranteed way to use the product.

``make chat`` starts a conversation; ``make ingest`` builds the index. Errors
are rendered as a message plus a suggested fix, never as a traceback.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from companion.chat.agent import CompanionAgent, ChatResponse
from companion.chat.providers import provider_available
from companion.chat.session import Session
from companion.config import get_settings
from companion.errors import CompanionError
from companion.ingest.pipeline import load_episodes, run_ingestion
from companion.utils.logging import configure_logging
from companion.utils.timefmt import format_duration

console = Console()

HELP_TEXT = """\
[bold]Commands[/bold]
  /episodes            list the episodes in this collection
  /episode <n|id>      scope questions to one episode (0 or 'all' clears it)
  /sources             show the passages behind the last answer
  /clear               start a fresh conversation
  /help                show this help
  /quit                exit

[bold]Things worth asking[/bold]
  What is the main idea of the Planck episode?
  Explain that more simply.
  How do these episodes differ in how they treat uncertainty?
  Which episode should I start with to understand entropy?
  Take me to the part where they explain the ultraviolet catastrophe.
  What do these episodes say about dark matter?      (correctly refused)
"""


def _print_episodes(episodes) -> None:
    table = Table(title="Episodes in this collection", title_style="bold",
                  header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Episode")
    table.add_column("Length", justify="right")
    table.add_column("Passages", justify="right", style="dim")
    for index, episode in enumerate(episodes, start=1):
        table.add_row(
            str(index), episode.title,
            format_duration(episode.duration), str(episode.num_chunks),
        )
    console.print(table)


def _print_answer(response: ChatResponse) -> None:
    console.print()
    if response.not_covered:
        console.print(Panel(response.answer, title="Not covered",
                            border_style="yellow", title_align="left"))
    else:
        console.print(Markdown(response.answer))

    if response.sources:
        console.print()
        console.print("[bold]Sources[/bold] [dim](verify these in the audio)[/dim]")
        for source in response.sources:
            console.print(
                f"  [cyan]•[/cyan] {source['episode_title']} "
                f"[bold]{source['timestamp']}[/bold] "
                f"[dim]({source['source_file']})[/dim]"
            )
    invalid = [c for c in response.citations if not c.valid]
    if invalid:
        console.print(
            f"  [dim yellow]{len(invalid)} citation(s) failed validation and "
            f"were removed.[/dim yellow]"
        )
    console.print(
        f"[dim]{response.mode} · {response.latency_ms / 1000:.1f}s"
        + (f" · ${response.estimated_cost_usd:.4f}" if response.llm else "")
        + "[/dim]"
    )
    console.print()


def _resolve_episode(argument: str, episodes) -> str | None:
    argument = argument.strip()
    if not argument or argument.lower() in {"0", "all", "none", "clear"}:
        return None
    if argument.isdigit():
        index = int(argument) - 1
        if 0 <= index < len(episodes):
            return episodes[index].episode_id
        raise ValueError(f"there is no episode {argument}")
    for episode in episodes:
        if argument.lower() in (episode.episode_id, episode.title.lower()):
            return episode.episode_id
    raise ValueError(f"no episode matches '{argument}'")


def run_chat() -> int:
    """Interactive REPL."""
    settings = get_settings()
    episodes = load_episodes(settings)
    if not episodes:
        console.print(
            "[yellow]No episodes have been ingested yet.[/yellow]\n"
            "  → Put the podcast audio in audio/ and run [bold]make ingest[/bold]."
        )
        return 1
    if not provider_available("chat", settings):
        console.print(
            f"[yellow]No API key found for CHAT_PROVIDER="
            f"{settings.chat_provider}.[/yellow]\n"
            "  → Copy .env.example to .env and add your key. Retrieval works "
            "without one: try [bold]make search Q=\"your question\"[/bold]."
        )
        return 1

    console.print(Panel.fit(
        "[bold]Fermi Podcast Companion[/bold]\n"
        "[dim]Answers come only from the supplied episodes, with timestamps "
        "you can check.[/dim]",
        border_style="cyan",
    ))
    _print_episodes(episodes)
    console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    agent = CompanionAgent(settings)
    session = Session()

    while True:
        try:
            message = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0
        if not message:
            continue

        if message.startswith("/"):
            command, _, argument = message[1:].partition(" ")
            command = command.lower()
            if command in {"quit", "exit", "q"}:
                console.print("[dim]Bye.[/dim]")
                return 0
            if command == "help":
                console.print(HELP_TEXT)
            elif command == "episodes":
                _print_episodes(episodes)
            elif command == "episode":
                try:
                    session.episode_filter = _resolve_episode(argument, episodes)
                except ValueError as exc:
                    console.print(f"[yellow]{exc}[/yellow]")
                    continue
                scope = session.episode_filter or "all episodes"
                console.print(f"[dim]Scoped to: {scope}[/dim]")
            elif command == "clear":
                session.clear()
                console.print("[dim]Conversation cleared.[/dim]")
            elif command == "sources":
                turn = session.last_turn
                if not turn or not turn.retrieved:
                    console.print("[dim]No passages retrieved yet.[/dim]")
                else:
                    for item in turn.retrieved:
                        console.print(
                            f"  [cyan]{item.chunk.episode_title}[/cyan] "
                            f"[bold]{item.chunk.timestamp_label}[/bold] "
                            f"[dim](score {item.score:.4f})[/dim]\n"
                            f"    {item.chunk.text[:220]}..."
                        )
            else:
                console.print(f"[yellow]Unknown command '/{command}'. "
                              f"Try /help.[/yellow]")
            continue

        try:
            with console.status("[dim]searching the episodes...[/dim]"):
                response = agent.ask(message, session)
        except CompanionError as exc:
            console.print(f"\n[red]{exc.message}[/red]")
            if exc.remedy:
                console.print(f"[yellow]  → {exc.remedy}[/yellow]\n")
            continue
        _print_answer(response)


def run_ingest(force: bool) -> int:
    configure_logging()
    settings = get_settings()
    stats = run_ingestion(settings, force=force)
    console.print()
    table = Table(title="Ingestion complete", header_style="bold cyan")
    table.add_column("Episode")
    table.add_column("Length", justify="right")
    table.add_column("Segments", justify="right")
    table.add_column("Chunks", justify="right")
    for episode in stats.episodes:
        table.add_row(
            episode.title, format_duration(episode.duration),
            str(episode.num_segments), str(episode.num_chunks),
        )
    console.print(table)
    console.print(
        f"[dim]audio {format_duration(stats.total_audio_seconds)} · "
        f"transcribed {len(stats.transcribed)} · reused {len(stats.reused)} · "
        f"ASR {format_duration(stats.asr_seconds)} · "
        f"total {format_duration(stats.total_seconds)}[/dim]"
    )
    console.print("\nNext: [bold]make chat[/bold] (or [bold]make web[/bold])")
    return 0


def run_search(query: str, top_k: int) -> int:
    """Retrieval-only search. Works with no API key — useful for debugging."""
    configure_logging()
    from companion.retrieve.retriever import get_retriever

    result = get_retriever().retrieve(query, top_k=top_k)
    if result.is_empty:
        console.print("[yellow]Nothing matched that query.[/yellow]")
        return 0
    console.print()
    for rank, item in enumerate(result.chunks, start=1):
        console.print(
            f"[bold]{rank}.[/bold] [cyan]{item.chunk.episode_title}[/cyan] "
            f"[bold]{item.chunk.timestamp_label}[/bold] "
            f"[dim]rrf={item.score:.4f} dense={item.dense_score} "
            f"bm25={item.bm25_score}[/dim]"
        )
        console.print(f"   {item.chunk.text[:260]}...\n")
    return 0


def run_episodes() -> int:
    episodes = load_episodes()
    if not episodes:
        console.print("[yellow]Nothing ingested yet. Run `make ingest`.[/yellow]")
        return 1
    _print_episodes(episodes)
    return 0


def run_models(filter_text: str | None, limit: int) -> int:
    """List models available through OpenRouter, cheapest first."""
    from companion.chat.providers import list_openrouter_models

    models = list_openrouter_models()

    def price(model: dict) -> tuple[float, float]:
        pricing = model.get("pricing") or {}
        try:
            return (
                float(pricing.get("prompt") or 0) * 1e6,
                float(pricing.get("completion") or 0) * 1e6,
            )
        except (TypeError, ValueError):
            return (0.0, 0.0)

    rows = []
    for model in models:
        model_id = model.get("id", "")
        if filter_text and filter_text.lower() not in model_id.lower():
            continue
        prompt_price, completion_price = price(model)
        rows.append((prompt_price, completion_price, model_id,
                     model.get("context_length")))
    rows.sort()

    table = Table(
        title=f"OpenRouter models ({len(rows)} shown of {len(models)})",
        header_style="bold cyan",
    )
    table.add_column("Model id")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    table.add_column("Context", justify="right", style="dim")
    for prompt_price, completion_price, model_id, context in rows[:limit]:
        table.add_row(
            model_id, f"{prompt_price:.2f}", f"{completion_price:.2f}",
            f"{context:,}" if context else "—",
        )
    console.print(table)
    console.print(
        "[dim]Set CHAT_MODEL / JUDGE_MODEL in .env to one of these ids.[/dim]"
    )
    return 0


def run_doctor() -> int:
    """Report on the environment so setup problems are self-diagnosing."""
    import shutil

    settings = get_settings()
    table = Table(title="Environment check", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    table.add_row("ffmpeg", "[green]ok[/green]" if ffmpeg_ok else "[red]missing[/red]",
                  shutil.which("ffmpeg") or "brew install ffmpeg")

    audio = sorted(settings.audio_dir.glob("*.mp3")) if settings.audio_dir.exists() else []
    table.add_row("audio files",
                  "[green]ok[/green]" if audio else "[yellow]none[/yellow]",
                  f"{len(audio)} file(s) in {settings.audio_dir}")

    episodes = load_episodes(settings)
    table.add_row("ingested", "[green]ok[/green]" if episodes else "[yellow]no[/yellow]",
                  f"{len(episodes)} episode(s); run `make ingest`" if not episodes
                  else f"{len(episodes)} episode(s)")

    index_ok = (settings.vector_db_path / "index_meta.json").exists()
    table.add_row("search index",
                  "[green]ok[/green]" if index_ok else "[yellow]not built[/yellow]",
                  str(settings.vector_db_path))

    chat_key = provider_available("chat", settings)
    table.add_row(f"chat key ({settings.chat_provider})",
                  "[green]ok[/green]" if chat_key else "[yellow]missing[/yellow]",
                  settings.chat_model)
    judge_key = provider_available("judge", settings)
    table.add_row(f"judge key ({settings.judge_provider})",
                  "[green]ok[/green]" if judge_key else "[yellow]missing[/yellow]",
                  settings.judge_model)

    console.print(table)

    if settings.chat_provider == "openrouter" and chat_key:
        _check_openrouter_models(settings)

    if not chat_key:
        key_name = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(settings.chat_provider, "the provider's API key")
        console.print(
            f"\n[yellow]Chat and evaluation need an API key.[/yellow]\n"
            f"  → add {key_name} to .env "
            f"(CHAT_PROVIDER={settings.chat_provider})"
        )
    return 0


def _check_openrouter_models(settings) -> None:
    """Warn early if CHAT_MODEL / JUDGE_MODEL are not real OpenRouter ids."""
    from companion.chat.providers import list_openrouter_models
    from companion.errors import CompanionError

    try:
        available = {model.get("id") for model in list_openrouter_models(settings)}
    except CompanionError as exc:
        console.print(f"[dim]Could not verify model ids: {exc.message}[/dim]")
        return
    for label, model in (
        ("CHAT_MODEL", settings.chat_model),
        ("JUDGE_MODEL", settings.judge_model),
    ):
        if model in available:
            console.print(f"[green]✓[/green] {label} [bold]{model}[/bold] "
                          f"is available on OpenRouter")
        else:
            console.print(
                f"[red]✗[/red] {label} [bold]{model}[/bold] was not found on "
                f"OpenRouter.\n  → run [bold]make models[/bold] to list valid ids"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="companion", description="Fermi Podcast Companion"
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="transcribe audio and build indexes")
    ingest.add_argument("--force", action="store_true",
                        help="re-transcribe even if nothing changed")
    subparsers.add_parser("chat", help="start a conversation")
    subparsers.add_parser("episodes", help="list ingested episodes")
    subparsers.add_parser("doctor", help="check the environment")
    models = subparsers.add_parser(
        "models", help="list models available through OpenRouter"
    )
    models.add_argument("--filter", dest="filter_text",
                        help="substring to match against model ids")
    models.add_argument("--limit", type=int, default=40)
    search = subparsers.add_parser("search", help="retrieval only, no API key needed")
    search.add_argument("query", nargs="+")
    search.add_argument("-k", "--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return run_ingest(args.force)
        if args.command == "episodes":
            return run_episodes()
        if args.command == "doctor":
            return run_doctor()
        if args.command == "models":
            return run_models(args.filter_text, args.limit)
        if args.command == "search":
            return run_search(" ".join(args.query), args.top_k)
        return run_chat()
    except CompanionError as exc:
        console.print(f"\n[red]{exc.message}[/red]")
        if exc.remedy:
            console.print(f"[yellow]  → {exc.remedy}[/yellow]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
