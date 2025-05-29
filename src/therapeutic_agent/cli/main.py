"""Advanced CLI interface with interactive session management."""

import asyncio
from datetime import datetime
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from therapeutic_agent.core.exceptions import TherapeuticAgentException
from therapeutic_agent.core.session_manager import TherapeuticSessionManager

app = typer.Typer(help="Therapeutic Agent CLI - Interactive therapeutic sessions")
console = Console()


@app.command()
def start_session(
    user_id: str = typer.Option(..., "--user", "-u", help="User identifier"),
    title: str = typer.Option(None, "--title", "-t", help="Session title"),
) -> None:
    """Start an interactive therapeutic session."""
    asyncio.run(_interactive_session(user_id, title))


@app.command()
def list_sessions(
    user_id: str = typer.Option(..., "--user", "-u", help="User identifier"),
) -> None:
    """List user's therapeutic sessions."""
    msg = "Session listing functionality would require "
    msg += "additional repository methods"
    console.print(f"[yellow]{msg}[/yellow]")


@app.command()
def session_info(
    session_id: str = typer.Argument(..., help="Session ID"),
    user_id: str = typer.Option(None, "--user", "-u", help="User identifier"),
) -> None:
    """Display information about a specific session."""
    asyncio.run(_show_session_info(UUID(session_id), user_id))


async def _interactive_session(user_id: str, title: str | None) -> None:
    """Run interactive therapeutic session."""
    session_manager = TherapeuticSessionManager()

    try:
        console.print("\n[bold green]Starting Therapeutic Session[/bold green]")
        console.print(
            "[dim]Type 'quit', 'exit', or 'end' to finish the session[/dim]\n"
        )

        session_data = await session_manager.create_session(user_id, title)
        session_id = UUID(session_data["session_id"])

        session_info_msg = "[bold]Session Created[/bold]\n"
        session_info_msg += f"ID: {session_data['session_id']}\n"
        session_info_msg += f"Title: {session_data.get('title', 'Untitled')}"
        console.print(
            Panel(
                session_info_msg,
                title="Session Info",
                border_style="green",
            )
        )

        while True:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.lower() in ["quit", "exit", "end"]:
                console.print("\n[yellow]Ending session...[/yellow]")
                break

            if not user_input.strip():
                console.print("[red]Please enter a message[/red]")
                continue

            try:
                console.print("[dim]Processing...[/dim]")
                response = await session_manager.send_message(
                    session_id, user_input, user_id
                )

                style = "red" if response.get("safety_intervention") else "green"
                title = (
                    "Safety Response"
                    if response.get("safety_intervention")
                    else "Therapist"
                )

                console.print(
                    Panel(
                        response["content"],
                        title=f"[bold]{title}[/bold]",
                        border_style=style,
                    )
                )

                if response.get("metadata", {}).get("processing_time_ms"):
                    processing_time = response["metadata"]["processing_time_ms"]
                    console.print(f"[dim]Response time: {processing_time}ms[/dim]")

            except TherapeuticAgentException as e:
                console.print(
                    Panel(
                        f"[red]Error: {e.message}[/red]",
                        title="System Error",
                        border_style="red",
                    )
                )
            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Unexpected error: {str(e)}[/red]",
                        title="System Error",
                        border_style="red",
                    )
                )

        try:
            end_result = await session_manager.end_session(session_id, user_id)
            summary_text = end_result.get("summary", "No summary available")
            summary_msg = "Session ended successfully\n"
            summary_msg += f"Messages: {end_result['message_count']}\n"
            summary_msg += f"Summary: {summary_text[:200]}..."
            console.print(
                Panel(
                    summary_msg,
                    title="Session Summary",
                    border_style="blue",
                )
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not properly end session: {str(e)}[/yellow]"
            )

    except TherapeuticAgentException as e:
        console.print(f"[red]Error: {e.message}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")


async def _show_session_info(session_id: UUID, user_id: str | None) -> None:
    """Show detailed session information."""
    session_manager = TherapeuticSessionManager()

    try:
        session_data = await session_manager.get_session(session_id, user_id)

        info_table = Table(title="Session Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Session ID", session_data["session_id"])
        info_table.add_row("User ID", session_data["user_id"])
        info_table.add_row("Title", session_data.get("title", "Untitled"))
        info_table.add_row("Status", session_data["status"])
        info_table.add_row("Message Count", str(session_data["message_count"]))
        info_table.add_row("Safety Score", f"{session_data['safety_score']:.2f}")
        info_table.add_row("Created", session_data["created_at"])
        info_table.add_row("Last Activity", session_data["last_activity"])

        console.print(info_table)

        if session_data.get("summary"):
            console.print(
                Panel(
                    session_data["summary"],
                    title="Session Summary",
                    border_style="blue",
                )
            )

        if session_data.get("messages"):
            msg_count = len(session_data["messages"])
            console.print(
                f"\n[bold]Conversation History ({msg_count} messages)[/bold]\n"
            )

            for msg in session_data["messages"][-10:]:  # Show last 10 messages
                timestamp = datetime.fromisoformat(
                    msg["timestamp"].replace("Z", "+00:00")
                )
                time_str = timestamp.strftime("%H:%M:%S")

                role_style = "blue" if msg["role"] == "user" else "green"
                role_name = "You" if msg["role"] == "user" else "Therapist"

                role_display = f"[{role_style}][bold]{role_name}[/bold]"
                console.print(f"{role_display}[/{role_style}] [{time_str}]")
                content = msg["content"][:200]
                ellipsis = "..." if len(msg["content"]) > 200 else ""
                console.print(f"{content}{ellipsis}\n")

    except TherapeuticAgentException as e:
        console.print(f"[red]Error: {e.message}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")


if __name__ == "__main__":
    app()
