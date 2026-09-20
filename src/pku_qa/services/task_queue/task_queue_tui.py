#!/usr/bin/env python3
"""Black terminal UI for the persistent task queue."""

from __future__ import annotations

import argparse
import curses
import shlex
from pathlib import Path

from task_queue_cli import stop_process
from task_queue_manager import DEFAULT_DB, ROOT, TaskStore, tail_text


def prompt(screen, label: str) -> str:
    height, width = screen.getmaxyx()
    curses.echo()
    screen.addstr(height - 1, 0, " " * (width - 1))
    screen.addstr(height - 1, 0, label[: width - 2])
    screen.refresh()
    value = screen.getstr(height - 1, min(len(label), width - 2)).decode(
        "utf-8", errors="replace"
    )
    curses.noecho()
    return value.strip()


def run(screen, store: TaskStore) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.timeout(1000)
    selected = 0
    message = ""
    while True:
        tasks = store.list_tasks()
        selected = max(0, min(selected, max(0, len(tasks) - 1)))
        screen.erase()
        height, width = screen.getmaxyx()
        title = " PKU TASK QUEUE "
        screen.attron(curses.A_REVERSE)
        screen.addstr(0, 0, title.ljust(width - 1))
        screen.attroff(curses.A_REVERSE)
        help_text = "j/k move  a add  e edit  u/d reorder  p pause  r retry  x cancel  l log  q quit"
        screen.addstr(1, 0, help_text[: width - 1])
        screen.addstr(2, 0, "ID           STATUS      RESTART   PROGRESS   NAME"[: width - 1])
        visible = max(1, height - 6)
        start = max(0, selected - visible + 1)
        for row_index, task in enumerate(tasks[start : start + visible]):
            index = start + row_index
            progress = task.to_dict().get("progress") or {}
            percent = progress.get("percent")
            progress_text = f"{percent:6.1f}%" if percent is not None else "     - "
            line = (
                f"{task.id:<12} {task.status:<11} "
                f"{task.restart_count:>2}/{task.max_restarts:<2} "
                f"{progress_text:<10} {task.name}"
            )
            if index == selected:
                screen.attron(curses.A_REVERSE)
            screen.addstr(3 + row_index, 0, line[: width - 1])
            if index == selected:
                screen.attroff(curses.A_REVERSE)
        screen.addstr(height - 2, 0, message[: width - 1])
        screen.refresh()
        key = screen.getch()
        message = ""
        if key in (ord("q"), 27):
            return
        if key in (ord("j"), curses.KEY_DOWN):
            selected = min(selected + 1, max(0, len(tasks) - 1))
        elif key in (ord("k"), curses.KEY_UP):
            selected = max(0, selected - 1)
        elif key == ord("a"):
            name = prompt(screen, "name: ")
            command = prompt(screen, "command: ")
            if name and command:
                store.add(
                    name=name,
                    command=shlex.split(command),
                    cwd=ROOT,
                )
                message = "task added"
        elif tasks and key == ord("e"):
            name = prompt(screen, f"name [{tasks[selected].name}]: ")
            if name:
                store.update(tasks[selected].id, name=name)
                message = "task updated"
        elif tasks and key in (ord("u"), ord("d")):
            destination = selected - 1 if key == ord("u") else selected + 1
            store.move(tasks[selected].id, destination)
            selected = max(0, min(destination, len(tasks) - 1))
        elif tasks and key == ord("p"):
            task = tasks[selected]
            store.update(task.id, status="paused")
            stop_process(task)
            store.update(task.id, status="paused", pid=None, runner_pid=None)
            message = "task paused"
        elif tasks and key == ord("r"):
            task = tasks[selected]
            store.update(task.id, status="paused")
            stop_process(task)
            store.update(
                task.id,
                status="queued",
                adopted=False,
                pid=None,
                runner_pid=None,
                next_retry_at=0,
                enabled=True,
            )
            message = "task queued for retry"
        elif tasks and key == ord("x"):
            task = tasks[selected]
            store.update(task.id, status="cancelled", enabled=False)
            stop_process(task)
            store.update(
                task.id,
                status="cancelled",
                enabled=False,
                pid=None,
                runner_pid=None,
            )
            message = "task cancelled"
        elif tasks and key == ord("l"):
            log = tail_text(tasks[selected].log_path, 8000).splitlines()
            screen.erase()
            for index, line in enumerate(log[-(height - 2) :]):
                screen.addstr(index, 0, line[: width - 1])
            screen.addstr(height - 1, 0, "press any key")
            screen.timeout(-1)
            screen.getch()
            screen.timeout(1000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    store = TaskStore(Path(args.db))
    curses.wrapper(run, store)


if __name__ == "__main__":
    main()
