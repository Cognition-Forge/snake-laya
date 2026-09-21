from collections import deque

import pytest
from helpers import make_board, serpentine

from snake_laya.game import DIR_ORDER, MIN_H, MIN_W, START_LEN, Board, Dir, collision, legal_dirs


@pytest.mark.parametrize(
    "d, opposite", [(Dir.UP, Dir.DOWN), (Dir.DOWN, Dir.UP), (Dir.LEFT, Dir.RIGHT), (Dir.RIGHT, Dir.LEFT)]
)
def test_dir_opposite(d, opposite):
    assert d.opposite is opposite


@pytest.mark.parametrize(
    "w, h, ok",
    [(MIN_W, MIN_H, True), (MIN_W - 1, MIN_H, False), (MIN_W, MIN_H - 1, False), (0, 0, False), (30, 20, True)],
)
def test_board_size_validation(w, h, ok):
    if ok:
        Board(w, h, seed=1)
    else:
        with pytest.raises(ValueError):
            Board(w, h, seed=1)


@pytest.mark.parametrize("w, h", [(MIN_W, MIN_H), (30, 20), (40, 30)])
def test_initial_state(w, h):
    b = Board(w, h, seed=3)
    assert len(b.body) == START_LEN
    assert b.heading is Dir.RIGHT
    assert b.alive and not b.cleared
    assert (b.score, b.total_food, b.best, b.deaths, b.steps) == (0, 0, 0, 0, 0)
    assert all(0 <= x < w and 0 <= y < h for x, y in b.body)
    assert b.food is not None and b.food not in b.body
    # room to move right at start
    assert b.collision(Dir.RIGHT) is None


def test_same_seed_same_initial_food_and_spawn_sequence():
    a, b = Board(30, 20, seed=42), Board(30, 20, seed=42)
    assert a.food == b.food
    for board in (a, b):
        board.body = deque([(5, 5), (4, 5), (3, 5)])
        board.heading = Dir.RIGHT
        board.food = (6, 5)
    assert a.step(Dir.RIGHT).ate and b.step(Dir.RIGHT).ate
    assert a.food == b.food


def test_food_spawn_is_row_major_over_free_cells():
    # Only one free cell left → spawn must pick it.
    cells = serpentine(MIN_W, MIN_H)
    board = make_board(cells[1:], Dir.LEFT, None)
    assert board._spawn_food() == cells[0]


@pytest.mark.parametrize(
    "body, heading, d, reason",
    [
        ([(0, 3), (1, 3), (2, 3)], Dir.LEFT, Dir.LEFT, "wall"),  # left edge
        ([(11, 3), (10, 3), (9, 3)], Dir.RIGHT, Dir.RIGHT, "wall"),  # right edge
        ([(5, 0), (5, 1), (5, 2)], Dir.UP, Dir.UP, "wall"),  # top edge
        ([(5, 7), (5, 6), (5, 5)], Dir.DOWN, Dir.DOWN, "wall"),  # bottom edge
        ([(2, 2), (3, 2), (3, 3), (2, 3), (1, 3)], Dir.LEFT, Dir.DOWN, "body"),  # turns into own body
        ([(5, 3), (4, 3), (3, 3)], Dir.RIGHT, Dir.RIGHT, None),  # open space
        ([(0, 3), (1, 3), (2, 3)], Dir.LEFT, Dir.UP, None),  # hugging wall is fine
    ],
)
def test_collision(body, heading, d, reason):
    assert collision(12, 8, body, None, d) == reason
    board = make_board(body, heading, (11, 7) if (11, 7) not in body else (0, 0))
    result = board.step(d)
    assert result.died is (reason is not None)
    assert board.alive is (reason is None)
    assert board.deaths == (1 if reason else 0)


# 2x2 loop: head (1,1) heading LEFT, tail (1,2) directly below the head.
LOOP = [(1, 1), (2, 1), (2, 2), (1, 2)]


@pytest.mark.parametrize(
    "food, expected",
    [
        ((9, 5), None),  # tail vacates this step → safe
        ((1, 2), "body"),  # eating keeps the tail in place → blocked (synthetic: food on tail cell)
    ],
)
def test_tail_chase(food, expected):
    assert collision(12, 8, LOOP, food, Dir.DOWN) == expected


def test_tail_chase_step_keeps_length():
    board = make_board(LOOP, Dir.LEFT, (9, 5))
    result = board.step(Dir.DOWN)
    assert not result.died and board.alive
    assert list(board.body) == [(1, 2), (1, 1), (2, 1), (2, 2)]


def test_reverse_is_illegal_not_fatal():
    board = make_board([(5, 3), (4, 3), (3, 3)], Dir.RIGHT, (9, 5))
    assert Dir.LEFT not in board.legal_dirs()
    with pytest.raises(ValueError):
        board.step(Dir.LEFT)
    assert board.alive and board.steps == 0


@pytest.mark.parametrize(
    "body, heading, expected",
    [
        ([(5, 5)], Dir.RIGHT, DIR_ORDER),  # length 1: reverse allowed
        ([(5, 5), (4, 5)], Dir.RIGHT, (Dir.UP, Dir.DOWN, Dir.RIGHT)),
        ([(5, 5), (5, 6)], Dir.UP, (Dir.UP, Dir.LEFT, Dir.RIGHT)),
    ],
)
def test_legal_dirs(body, heading, expected):
    assert legal_dirs(body, heading) == expected


def test_eating_grows_scores_and_respawns_food():
    board = make_board([(5, 3), (4, 3), (3, 3)], Dir.RIGHT, (6, 3))
    result = board.step(Dir.RIGHT)
    assert result.ate and not result.died
    assert len(board.body) == 4 and board.head == (6, 3)
    assert (board.score, board.total_food, board.best) == (1, 1, 1)
    assert board.food is not None and board.food not in board.body


def test_death_then_step_raises_and_respawn_keeps_totals():
    board = make_board([(11, 3), (10, 3), (9, 3)], Dir.RIGHT, (0, 0))
    board.score, board.total_food, board.best = 4, 7, 5
    assert board.step(Dir.RIGHT).died
    assert board.best == 5
    with pytest.raises(RuntimeError):
        board.step(Dir.UP)
    board.respawn()
    assert board.alive and board.score == 0 and len(board.body) == START_LEN
    assert (board.total_food, board.best, board.deaths) == (7, 5, 1)
    assert board.food not in board.body


def test_respawn_moves_food_off_new_body():
    board = Board(MIN_W, MIN_H, seed=1)
    start_cells = list(board.body)
    board.alive = False
    board.food = start_cells[2]  # would overlap the respawned snake
    board.respawn()
    assert board.food is not None and board.food not in board.body


def test_board_full_clears_and_stops():
    cells = serpentine(MIN_W, MIN_H)
    board = make_board(cells[1:], Dir.LEFT, cells[0])
    result = board.step(Dir.LEFT)
    assert result.ate
    assert board.cleared and board.food is None and not board.active
    with pytest.raises(RuntimeError):
        board.step(Dir.DOWN)


def test_view_is_immutable_snapshot():
    board = Board(30, 20, seed=5)
    view = board.view()
    board.step(Dir.RIGHT)
    assert view.body != tuple(board.body)
    assert view.head == view.body[0] and view.length == START_LEN
