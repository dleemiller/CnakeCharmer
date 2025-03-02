from celery import shared_task
import os
import logging
from cnake_charmer.generate.code_generator import CodeGenerator
from cnake_charmer.generate.database import CodeDatabase
from cnake_charmer.generate.equivalency_checker import EquivalencyChecker
from cnake_charmer.generate.ephemeral_runner.builders import get_builder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("celery_tasks")

# Get database URL from environment or use default
db_url = os.environ.get("DATABASE_URL", "postgresql://user:password@db/cnake_charmer")
db = CodeDatabase(db_url)

# Get OpenRouter API key from environment
api_key = os.environ.get("OPENROUTER_API_KEY")
generator = CodeGenerator(api_key=api_key)

checker = EquivalencyChecker("/usr/bin/python3", "/usr/bin/cython")

@shared_task(name="generate_code_task")
def generate_code_task(prompt: str):
    """
    Celery task to generate Python & Cython code from a prompt.
    """
    logger.info(f"Starting code generation for prompt: {prompt}")
    
    try:
        # Generate code
        generated_code = generator.generate_equivalent_code(prompt_id=prompt, language="python")
        
        # Save to database
        db.save_code(prompt, generated_code, build_status="generated")
        
        logger.info("Code generated and saved successfully")
        return {"message": "Code generated successfully!", "generated_code": generated_code}
    except Exception as e:
        logger.error(f"Error in generate_code_task: {e}")
        # Return error message but don't raise (to avoid retries)
        return {"error": str(e)}

@shared_task(name="check_build_status_task")
def check_build_status_task(task_id: str):
    """
    Celery task to check if the generated code builds successfully.
    """
    logger.info(f"Checking build status for task: {task_id}")
    
    try:
        code_entry = db.get_code(task_id)
        if not code_entry:
            error_msg = f"No code found for task ID: {task_id}"
            logger.error(error_msg)
            return {"error": error_msg}
            
        builder = get_builder(code_entry["language"])
        error = builder.build_and_run(code_entry["code"])
        
        if error:
            logger.warning(f"Build failed for task {task_id}: {error}")
            db.update_status(task_id, "failed")
            return {"error": error}
        
        logger.info(f"Build successful for task: {task_id}")
        db.update_status(task_id, "built")
        return {"message": "Build successful!"}
    except Exception as e:
        logger.error(f"Error in check_build_status_task: {e}")
        return {"error": str(e)}

@shared_task(name="execute_task")
def execute_task(task_id: str):
    """
    Celery task to execute code and check equivalency.
    """
    logger.info(f"Executing code for task: {task_id}")
    
    try:
        code_entry = db.get_code(task_id)
        if not code_entry:
            error_msg = f"No code found for task ID: {task_id}"
            logger.error(error_msg)
            return {"error": error_msg}
            
        if not checker.check_equivalency():
            error_msg = "Equivalency test failed!"
            logger.warning(f"{error_msg} for task: {task_id}")
            return {"error": error_msg}

        logger.info(f"Code executed successfully for task: {task_id}")
        db.update_status(task_id, "executed")
        return {"message": "Code executed successfully!"}
    except Exception as e:
        logger.error(f"Error in execute_task: {e}")
        return {"error": str(e)}