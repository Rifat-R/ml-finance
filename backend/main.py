from typing import Union
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/random")
def read_random():
    sample_messages = [
        "The market rewards patience.",
        "Diversification keeps surprises manageable.",
        "Volatility is a feature, not a bug.",
        "Probability beats prediction.",
        "Stay curious; trends change fast.",
    ]
    selected = random.choice(sample_messages)
    return {"message": selected}
