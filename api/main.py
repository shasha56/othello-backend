from fastapi import FastAPI
from pydantic import BaseModel
from .app_service import reset_game, play_turn, get_board, is_move, valid_actions
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
        status, boards, next_valid_actions = play_turn(request.action)
        boards = [board.flatten().tolist() for board in boards]
    else:
        status = "failed"
        boards = [get_board().board.flatten().tolist()]
        next_valid_actions = valid_actions()

    return {"status": status, "boards": boards, "next_actions": next_valid_actions}

@app.get("/board")
def board():
    return {"board": get_board().board.flatten().tolist(), "next_actions": valid_actions()}

@app.post("/check")
def check_moved(request: MoveRequest):
    if is_move(request.action):
        return {"status": True}
    else:
        return {"status": False}