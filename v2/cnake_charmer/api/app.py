# api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import generate, status, results


app = FastAPI(
    title="CnakeCharmer API",
    description="API for generating, analyzing, and optimizing code in multiple languages",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generate.router)
app.include_router(status.router)
app.include_router(results.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)