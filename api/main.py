from fastapi import FastAPI
from pydantic import BaseModel
from .app_service import reset_game, play_turn, get_board

app = FastAPI()

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
    play_turn(request.action)
    return {"message": "move"}

@app.get("/board")
def board():
    return {"board": get_board().board.tolist()}