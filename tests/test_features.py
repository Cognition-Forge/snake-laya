import pytest
from helpers import make_view

from snake_laya.features import (
    INSTRUCTIONS,
    Move,
    analyse,
    build_question,
    build_state,
    criterion,
    flood_fill,
    food_text,
)
from snake_laya.game import Dir

STRAIGHT = [(5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]  # head (5,3) heading right, 12x8


@pytest.mark.parametrize(
    "head, food, text",
    [
        ((5, 5), (1, 3), "4 cells left, 2 cells up"),
        ((5, 5), (5, 8), "3 cells down"),
        ((5, 5), (6, 5), "1 cell right"),
        ((5, 5), (5, 4), "1 cell up"),
        ((5, 5), (6, 6), "1 cell right, 1 cell down"),
        ((5, 5), None, "none (board full)"),
    ],
)
def test_food_text(head, food, text):
    assert food_text(head, food) == text


@pytest.mark.parametrize(
    "move, text",
    [
        (Move(Dir.UP, "safe", None, 6, 5, 212, False), "moves toward the food"),
        (Move(Dir.DOWN, "safe", None, 6, 7, 212, False), "moves away from the food"),
        (Move(Dir.LEFT, "safe", None, 1, 0, 212, False), "eats the food"),
        (Move(Dir.LEFT, "safe", None, 1, 0, 4, True), "eats the food; leads into a dead end"),
        (Move(Dir.LEFT, "safe", None, 2, 3, 1, True), "moves away from the food; leads into a dead end"),
        (Move(Dir.LEFT, "safe", None, None, None, 30, False), "moves safely"),  # no food (board full)
        (Move(Dir.RIGHT, "fatal", "wall", 6, None, 0, False), "crashes into the wall"),
        (Move(Dir.RIGHT, "fatal", "body", 6, None, 0, False), "crashes into its own body"),
    ],
)
def test_criterion_templates(move, text):
    assert criterion(move) == text


def test_analyse_open_board_excludes_reverse():
    view = make_view(STRAIGHT, Dir.RIGHT, food=(9, 3))
    moves = analyse(view)
    assert list(moves) == [Dir.UP, Dir.DOWN, Dir.RIGHT]
    right = moves[Dir.RIGHT]
    assert (right.status, right.dist_before, right.dist_after) == ("safe", 4, 3)
    # 96 cells − 6 body cells after moving (tail vacated) = 90, all connected
    assert right.reachable == 90 and not right.trap
    assert moves[Dir.UP].dist_after == 5


@pytest.mark.parametrize(
    "body, heading, d, reason",
    [
        ([(11, 3), (10, 3), (9, 3)], Dir.RIGHT, Dir.RIGHT, "wall"),
        ([(0, 0), (1, 0), (2, 0)], Dir.LEFT, Dir.UP, "wall"),  # corner: two walls
        ([(0, 0), (1, 0), (2, 0)], Dir.LEFT, Dir.LEFT, "wall"),
        ([(2, 2), (3, 2), (3, 3), (2, 3), (1, 3)], Dir.LEFT, Dir.DOWN, "body"),
    ],
)
def test_analyse_fatal(body, heading, d, reason):
    moves = analyse(make_view(body, heading, food=(6, 6)))
    m = moves[d]
    assert (m.status, m.reason, m.reachable, m.dist_after) == ("fatal", reason, 0, None)


def pocket_body(k: int):
    """Body walls off pocket x∈{0,1}, y∈{0..k}; head (2,0) heading up; moving LEFT enters the pocket.

    Length = k+5; pocket cells reachable after entering = 2(k+1) − 1 = 2k+1.
    """
    column = [(2, y) for y in range(0, k + 2)]
    return [*column, (1, k + 1), (0, k + 1), (0, k + 2)]


@pytest.mark.parametrize(
    "k, reachable, trap",
    [
        (3, 7, True),  # 7 < length 8
        (4, 9, False),  # boundary: reachable == length → not a trap
        (5, 11, False),
    ],
)
def test_trap_boundary(k, reachable, trap):
    body = pocket_body(k)
    moves = analyse(make_view(body, Dir.UP, food=(11, 7)))
    left = moves[Dir.LEFT]
    assert left.safe
    assert (left.reachable, left.trap) == (reachable, trap)
    assert len(body) == k + 5


def test_eating_move_counts_tail_as_blocked():
    # Eating keeps the tail, so reachable is one less than the same move without food.
    body = STRAIGHT
    no_food = analyse(make_view(body, Dir.RIGHT, food=(11, 7)))[Dir.RIGHT].reachable
    eat = analyse(make_view(body, Dir.RIGHT, food=(6, 3)))[Dir.RIGHT].reachable
    assert eat == no_food - 1


@pytest.mark.parametrize(
    "blocked, start, expected",
    [
        (set(), (0, 0), 12 * 8 - 1),
        ({(1, 0), (0, 1)}, (0, 0), 0),  # boxed in a corner
        ({(0, 0), (1, 0), (0, 1)}, (0, 0), 0),
        ({(x, 1) for x in range(12)}, (0, 0), 11),  # sealed top row
    ],
)
def test_flood_fill(blocked, start, expected):
    assert flood_fill(12, 8, blocked, start) == expected


def test_build_state_and_question():
    view = make_view(STRAIGHT, Dir.RIGHT, food=(1, 1))
    state = build_state(view)
    assert state == {"game": "snake: eat the food, never crash", "food": "4 cells left, 2 cells up"}
    assert "right" not in state["game"]  # no heading: it biases Laya toward continuing straight
    q = build_question(analyse(view))["move"]
    assert q["type"] == "choice" and q["instructions"] == INSTRUCTIONS
    assert list(q["criteria"]) == ["up", "down", "right"]
    assert q["criteria"] == {
        "up": "moves toward the food",
        "down": "moves away from the food",
        "right": "moves away from the food",
    }


def test_single_free_cell_board():
    from helpers import serpentine

    cells = serpentine(12, 8)
    view = make_view(cells[1:], Dir.LEFT, food=cells[0])
    moves = analyse(view)
    assert moves[Dir.LEFT].safe and moves[Dir.LEFT].dist_after == 0
    assert all(not m.safe for d, m in moves.items() if d is not Dir.LEFT)
