"""pygame rendering: two side-by-side panels (human | computer), stats, overlays. Read-only over state."""

from __future__ import annotations

import pygame

from .config import Mode
from .game import DIR_ORDER, BoardView, Dir
from .match import Match, Phase
from .runner import RunnerView
from .stats import StatsView

FONT_STACK = "menlo,consolas,dejavusansmono,couriernew,monospace"

KEY_DIRS = {pygame.K_UP: Dir.UP, pygame.K_DOWN: Dir.DOWN, pygame.K_LEFT: Dir.LEFT, pygame.K_RIGHT: Dir.RIGHT}

BG = (5, 10, 20)
PANEL = (8, 15, 28)
BORDER = (24, 36, 56)
BOARD_BG = (6, 12, 24)
DIM = (92, 108, 134)
TEXT = (214, 224, 238)
HEAD = (236, 246, 255)
FOOD = (245, 197, 107)
DANGER = (232, 110, 110)
TRACK = (22, 32, 50)
HUMAN_ACCENT = (110, 214, 172)
HUMAN_TAIL = (40, 96, 82)
LAYA_ACCENT = (146, 166, 240)
LAYA_TAIL = (58, 72, 128)

PAD = 24
HEADER_H = 64
STATS_H = 200
FOOTER_H = 48
GAP = 12
MARGIN = 12


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def fmt_num(value: float | None, spec: str = ".1f") -> str:
    return "—" if value is None else format(value, spec)


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def fmt_clock(seconds: float) -> str:
    s = int(seconds + 0.999)  # ceil: show 0:00 only when time is really up
    return f"{s // 60}:{s % 60:02d}"


class Renderer:
    def __init__(self, width: int, height: int):
        if not pygame.font.get_init():
            pygame.font.init()
        self.grid = (width, height)
        self.cell = max(8, min(28, 640 // width, 440 // height))
        self.board_w, self.board_h = width * self.cell, height * self.cell
        self.panel_w = self.board_w + 2 * PAD
        self.panel_h = HEADER_H + self.board_h + STATS_H
        self.size = (2 * self.panel_w + GAP + 2 * MARGIN, self.panel_h + FOOTER_H + 2 * MARGIN)
        self.f_huge = pygame.font.SysFont(FONT_STACK, 34, bold=True)
        self.f_big = pygame.font.SysFont(FONT_STACK, 26)
        self.f_mid = pygame.font.SysFont(FONT_STACK, 16)
        self.f_small = pygame.font.SysFont(FONT_STACK, 12)
        self.f_count = pygame.font.SysFont(FONT_STACK, 96, bold=True)

    # ---- primitives ------------------------------------------------------------------------

    def _text(self, surf, font, text, color, pos, anchor="topleft") -> pygame.Rect:
        img = font.render(text, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def _metric(self, surf, x, y, label, value, unit, accent) -> None:
        self._text(surf, self.f_small, label, DIM, (x, y))
        r = self._text(surf, self.f_big, value, accent, (x, y + 16))
        if unit:
            self._text(surf, self.f_small, unit, DIM, (r.right + 4, r.bottom - 4), "bottomleft")

    def _kv(self, surf, x, y, w, label, value, color=TEXT) -> None:
        self._text(surf, self.f_small, label, DIM, (x, y))
        self._text(surf, self.f_small, value, color, (x + w, y), "topright")

    # ---- frame -----------------------------------------------------------------------------

    def draw(
        self,
        surf: pygame.Surface,
        match: Match,
        runner: RunnerView | None,
        stats: StatsView,
        labels: dict[str, str],
    ) -> None:
        surf.fill(BG)
        left = pygame.Rect(MARGIN, MARGIN, self.panel_w, self.panel_h)
        right = pygame.Rect(left.right + GAP, MARGIN, self.panel_w, self.panel_h)

        human = match.human
        self._panel(
            surf,
            left,
            human.board.view(),
            "YOU · ARROW KEYS",
            HUMAN_ACCENT,
            HUMAN_TAIL,
            dead_note=human.respawn_at is not None,
        )
        self._human_stats(surf, left, match)

        mode_label = "SYNC" if match.ranked else f"{match.unranked_reason} · NOT RANKED"
        title = f"{labels.get('model', '')} · {labels.get('device', '')} · {mode_label}".upper()
        if runner is not None:
            self._panel(surf, right, runner.board, title, LAYA_ACCENT, LAYA_TAIL, dead_note=not runner.board.alive)
        else:
            self._panel_frame(surf, right, title)
        self._computer_stats(surf, right, runner, stats)

        self._footer(surf, match, runner)
        self._overlay(surf, match, runner, stats)

    def _panel_frame(self, surf, rect, title) -> pygame.Rect:
        pygame.draw.rect(surf, PANEL, rect, border_radius=6)
        pygame.draw.rect(surf, BORDER, rect, 1, border_radius=6)
        board = pygame.Rect(rect.x + PAD, rect.y + HEADER_H, self.board_w, self.board_h)
        pygame.draw.rect(surf, BOARD_BG, board)
        pygame.draw.rect(surf, BORDER, board.inflate(2, 2), 1)
        self._text(surf, self.f_small, title, DIM, (rect.right - PAD, rect.y + 10), "topright")
        return board

    def _panel(self, surf, rect, view: BoardView, title, accent, tail, dead_note: bool) -> None:
        board = self._panel_frame(surf, rect, title)
        x, y = rect.x + PAD, rect.y + 14
        r = self._text(surf, self.f_huge, f"{view.score:03d}", accent, (x, y))
        self._text(surf, self.f_small, "SCORE", DIM, (r.right + 6, r.bottom - 8), "bottomleft")
        info = f"LENGTH {view.length:03d}   TOTAL {view.total_food:03d}   BEST {view.best:03d}   DEATHS {view.deaths}"
        self._text(surf, self.f_small, info, DIM, (r.right + 64, r.bottom - 8), "bottomleft")

        c = self.cell
        if view.food is not None:
            fx, fy = view.food
            pygame.draw.circle(surf, FOOD, (board.x + fx * c + c // 2, board.y + fy * c + c // 2), max(3, c * 3 // 10))
        n = max(1, view.length - 1)
        for i, (bx, by) in enumerate(view.body):
            color = HEAD if i == 0 else lerp(accent, tail, i / n)
            if not view.alive:
                color = lerp(color, (60, 64, 76), 0.6)
            pygame.draw.rect(surf, color, (board.x + bx * c, board.y + by * c, c, c))
        if view.cleared:
            self._text(surf, self.f_mid, "BOARD CLEARED", accent, board.center, "center")
        elif dead_note and not view.alive:
            self._text(surf, self.f_mid, "CRASHED · RESPAWNING", DANGER, board.center, "center")

    def _stats_origin(self, rect) -> tuple[int, int]:
        return rect.x + PAD, rect.y + HEADER_H + self.board_h + 18

    def _human_stats(self, surf, rect, match: Match) -> None:
        x, y = self._stats_origin(rect)
        col = self.board_w // 4
        hv = match.human.stats.view()
        view = match.human.board.view()
        self._metric(surf, x, y, "MOVES", str(hv.moves), "", HUMAN_ACCENT)
        self._metric(surf, x + col, y, "TURNS", str(hv.turns), "", HUMAN_ACCENT)
        self._metric(surf, x + 2 * col, y, "TURNS / S", f"{hv.turns_per_s:.1f}", "", HUMAN_ACCENT)
        self._metric(surf, x + 3 * col, y, "TICK", str(match.cfg.tick_ms), "ms", HUMAN_ACCENT)
        y2 = y + 70
        pygame.draw.line(surf, BORDER, (x, y2 - 12), (x + self.board_w, y2 - 12))
        half = self.board_w // 2 - 16
        queued = " ".join(d.label.upper() for d in match.human.queue.pending) or "—"
        rows = [("HEADING", view.heading.label.upper()), ("QUEUED", queued), ("LENGTH", str(view.length))]
        for i, (k, v) in enumerate(rows):
            self._kv(surf, x, y2 + i * 22, half, k, v)
        help_rows = ["ARROWS  move", "SPACE   pause", "R       restart", "M       switch mode"]
        for i, line in enumerate(help_rows):
            self._text(surf, self.f_small, line, DIM, (x + half + 32, y2 + i * 22))

    def _computer_stats(self, surf, rect, runner: RunnerView | None, s: StatsView) -> None:
        x, y = self._stats_origin(rect)
        col = self.board_w // 4
        self._metric(surf, x, y, "PREDICT", fmt_num(s.last_ms), "ms", LAYA_ACCENT)
        self._metric(surf, x + col, y, "P50 LATENCY", fmt_num(s.p50), "ms", LAYA_ACCENT)
        self._metric(surf, x + 2 * col, y, "DECISIONS / S", f"{s.dps:.1f}", "", LAYA_ACCENT)
        self._metric(surf, x + 3 * col, y, "DECISIONS", str(s.decisions), "", LAYA_ACCENT)
        y2 = y + 70
        pygame.draw.line(surf, BORDER, (x, y2 - 12), (x + self.board_w, y2 - 12))

        half = self.board_w // 2 - 16
        probs = runner.probs if runner is not None else None
        track_x, track_w = x + 52, half - 52 - 44
        for i, d in enumerate(DIR_ORDER):
            ry = y2 + i * 22
            self._text(surf, self.f_small, d.label.upper(), DIM, (x, ry))
            pygame.draw.rect(surf, TRACK, (track_x, ry + 3, track_w, 9))
            if probs is not None and d in probs:
                p = probs[d]
                if p > 0:
                    pygame.draw.rect(surf, LAYA_ACCENT, (track_x, ry + 3, max(2, round(track_w * p)), 9))
                label = f"{p:.2f}"
            else:
                label = "—"  # reverse is never offered; late step has no probabilities
            self._text(surf, self.f_small, label, TEXT, (x + half, ry), "topright")

        kx = x + half + 32
        kw = self.board_w - half - 32
        executed = runner.executed.label.upper() if runner is not None and runner.executed else "—"
        if runner is not None and runner.last_late:
            executed += " (LATE)"
        rows = [
            ("P95 / ms", fmt_num(s.p95)),
            ("EXECUTED", executed),
            ("SHARPNESS", fmt_num(s.sharpness, ".2f")),
            ("OVERRIDES", str(s.overrides)),
            ("LATE", f"{s.late} ({fmt_pct(s.late_pct)})"),
            ("BASELINE AGREE", fmt_pct(s.agree_pct)),
        ]
        for i, (k, v) in enumerate(rows):
            self._kv(surf, kx, y2 - 4 + i * 18, kw, k, v)

    def _footer(self, surf, match: Match, runner: RunnerView | None) -> None:
        y = MARGIN + self.panel_h + 14
        w = self.size[0]
        left = f"{fmt_clock(match.remaining_s)} LEFT   ·   {str(match.mode).upper()}   ·   SEED {match.cfg.seed}"
        self._text(surf, self.f_mid, left, TEXT, (MARGIN + 4, y))
        if not match.ranked:
            standing = "NOT RANKED"
        elif runner is None:
            standing = ""
        else:
            standing = self._standing_text(match, runner.board, final=False)
        self._text(surf, self.f_mid, standing, TEXT, (w - MARGIN - 4, y), "topright")

    def _standing_text(self, match: Match, computer: BoardView, final: bool) -> str:
        who = match.standing(computer)
        human = match.human.board.view()
        margin = abs(human.total_food - computer.total_food)
        if who == "draw":
            return "DRAW" if final else "LEVEL"
        name = "YOU" if who == "human" else "LAYA"
        if final:
            return "YOU WIN" if who == "human" else "LAYA WINS"
        return f"LEADER: {name} +{margin}" if margin else f"LEADER: {name} (TIE-BREAK)"

    # ---- overlays --------------------------------------------------------------------------

    def _overlay(self, surf, match: Match, runner: RunnerView | None, s: StatsView) -> None:
        phase = match.phase
        if phase is Phase.RUNNING:
            return
        shade = pygame.Surface(self.size, pygame.SRCALPHA)
        shade.fill((2, 6, 14, 170))
        surf.blit(shade, (0, 0))
        center = (self.size[0] // 2, self.size[1] // 2)

        if phase is Phase.COUNTDOWN:
            self._text(surf, self.f_count, str(max(1, match.countdown_left)), TEXT, center, "center")
            return
        if phase is Phase.LOADING:
            self._box(surf, ["LOADING LAYA", "", "first run of a checkpoint downloads it", "ESC to quit"])
        elif phase is Phase.PAUSED:
            self._box(surf, ["PAUSED", "", "SPACE resume   ·   R restart   ·   M mode   ·   ESC quit"])
        elif phase is Phase.CONFIRM_MODE:
            if match.mode.other is Mode.MAX:
                target = "MAX SPEED (unranked)"
            else:
                target = "SYNC (ranked)" if match.cfg.equal_ticks else "SYNC (unranked: CPU tick differs)"
            lines = [f"SWITCH TO {target.upper()}?", "", "this restarts the match", "Y confirm   ·   N / ESC cancel"]
            self._box(surf, lines)
        elif phase is Phase.ERROR:
            lines = ["ERROR", "", *_wrap(match.error or "unknown error", 64), "", "ESC to quit"]
            self._box(surf, lines, title_color=DANGER)
        elif phase is Phase.OVER:
            self._box(surf, self._result_lines(match, runner, s))

    def _result_lines(self, match: Match, runner: RunnerView | None, s: StatsView) -> list[str]:
        human = match.human.board.view()
        if runner is None:
            return ["MATCH OVER"]
        comp = runner.board
        if match.ranked:
            title = self._standing_text(match, comp, final=True)
        else:
            title = f"NOT RANKED · {match.unranked_reason}"
        row = "{:<16}{:>8}{:>8}"
        return [
            title,
            "",
            row.format("", "YOU", "LAYA"),
            row.format("TOTAL FOOD", human.total_food, comp.total_food),
            row.format("BEST LIFE", human.best, comp.best),
            row.format("DEATHS", human.deaths, comp.deaths),
            "",
            f"decisions {s.decisions}   mean {fmt_num(s.mean_dps)}/s   stale {s.stale}",
            f"P50 {fmt_num(s.p50)} ms   P95 {fmt_num(s.p95)} ms",
            f"overrides {fmt_pct(s.override_pct)}   late {fmt_pct(s.late_pct)}",
            f"baseline agree {fmt_pct(s.agree_pct)}   sharpness {fmt_num(s.mean_sharpness, '.2f')}",
            "",
            "R restart   ·   M switch mode   ·   ESC quit",
        ]

    def _box(self, surf, lines: list[str], title_color=TEXT) -> None:
        imgs = [self.f_mid.render(line, True, title_color if i == 0 else TEXT) for i, line in enumerate(lines)]
        w = max(img.get_width() for img in imgs) + 64
        h = sum(max(img.get_height(), 8) for img in imgs) + 48
        box = pygame.Rect(0, 0, w, h)
        box.center = (self.size[0] // 2, self.size[1] // 2)
        pygame.draw.rect(surf, PANEL, box, border_radius=8)
        pygame.draw.rect(surf, BORDER, box, 1, border_radius=8)
        y = box.y + 24
        for img in imgs:
            surf.blit(img, img.get_rect(midtop=(box.centerx, y)))
            y += max(img.get_height(), 8)


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        line = ""
        for word in para.split():
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out
