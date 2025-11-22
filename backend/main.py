from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.messages import router as messages_router
from routers.predictor import router as predictor_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root
@app.get("/")
def read_root():
    return {"Hello": "World"}


# Include your external router
app.include_router(messages_router)
app.include_router(predictor_router)
