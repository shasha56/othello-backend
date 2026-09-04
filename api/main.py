from fastapi import FastAPI
from pydantic import BaseModel
from .app_service import reset_game, play_turn, get_board, is_move, valid_actions, get_turn, count_stone
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    action: int

@app.get("/")
def root():
    return {"message": "Othello API"}

@app.post("/reset")
def reset():
    reset_game()

    return {"message": "reset"}

@app.post("/move")
def move(request: MoveRequest):
    if is_move(request.action):
        status, boards, next_valid_actions, turns, stone_count = play_turn(request.action)
        boards = [board.flatten().tolist() for board in boards]
    else:
        status = "failed"
        boards = [get_board().board.flatten().tolist()]
        next_valid_actions = valid_actions()
        turns = get_turn()
        stone_count = [(count_stone())]

    return {"status": status, "boards": boards, "next_actions": next_valid_actions, "turns": turns, "stone_count": stone_count}

@app.get("/board")
def board():
    stone_count = [(count_stone())]
    return {"board": get_board().board.flatten().tolist(), "next_actions": valid_actions(), "turns": get_turn(), "stone_count": stone_count}

@app.post("/check")
def check_moved(request: MoveRequest):
    if is_move(request.action):
        return {"status": True}
    else:
        return {"status": False}