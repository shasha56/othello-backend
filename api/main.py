from fastapi import FastAPI
from pydantic import BaseModel
from .app_service import reset_game, play_turn, get_board
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
    status, boards = play_turn(request.action)
    boards = [board.tolist() for board in boards]

    return {"status": status, "boards": boards}

@app.get("/board")
def board():
    return {"board": get_board().board.tolist()}