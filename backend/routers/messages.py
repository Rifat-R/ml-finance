import random
from fastapi import APIRouter

router = APIRouter()


@router.get("/random")
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
