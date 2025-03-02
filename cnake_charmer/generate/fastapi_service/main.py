from fastapi import FastAPI
import os
import logging
from cnake_charmer.generate.fastapi_service.routes import router
import dspy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Create FastAPI app
app = FastAPI(
    title="CnakeCharmer Code Generator API",
    description="Generates, compiles, and validates Python & Cython code.",
    version="1.0"
)

# Include API routes
app.include_router(router)

# Configure DSPy on startup
@app.on_event("startup")
async def startup_event():
    """Initialize components when the application starts."""
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY is not set. API calls will likely fail.")
            
        # Configure DSPy with OpenRouter/Claude
        lm = dspy.LM(
            model="openrouter/anthropic/claude-3.7-sonnet", 
            cache=False, 
            max_tokens=2500,
            api_key=api_key
        )
        dspy.configure(lm=lm)
        logger.info("DSPy configured with OpenRouter/Claude")
        
        # Initialize database (could be a separate function if needed)
        from cnake_charmer.generate.database import CodeDatabase
        db_url = os.environ.get("DATABASE_URL", "postgresql://user:password@db/cnake_charmer")
        db = CodeDatabase(db_url)
        logger.info("Database connection initialized")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't re-raise, let the application start anyway

@app.get("/")
async def root():
    return {"message": "CnakeCharmer API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "ok",
            # You could add more detailed health check info here
        }
    }