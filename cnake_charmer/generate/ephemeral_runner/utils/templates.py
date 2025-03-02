"""
Template loading and rendering utilities.

This module provides functions for loading and rendering templates.
"""

import os
import logging
import pkg_resources
from typing import Dict, Any

# Configure logger
logger = logging.getLogger("ephemeral_runner.utils.templates")

def load_template(template_name: str) -> str:
    """
    Load a template from the templates directory.
    
    Args:
        template_name: Name of the template to load
        
    Returns:
        Template content as a string
    """
    try:
        # Try to load from package resources first
        template_path = f'templates/{template_name}'
        template_content = pkg_resources.resource_string(
            'ephemeral_runner', template_path
        ).decode('utf-8')
        logger.debug(f"Loaded template {template_name} from package resources")
        return template_content
    except Exception as e:
        logger.debug(f"Could not load template from package resources: {str(e)}")
        
        # Fall back to loading from current directory structure
        try:
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(module_dir, 'templates', template_name)
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            logger.debug(f"Loaded template {template_name} from file: {template_path}")
            return template_content
        except Exception as e2:
            logger.error(f"Failed to load template {template_name}: {str(e2)}")
            raise ValueError(f"Could not load template {template_name}: {str(e2)}")

def render_template(template_name: str, context: Dict[str, Any] = None) -> str:
    """
    Load and render a template with the given context.
    
    Args:
        template_name: Name of the template to render
        context: Dictionary of context variables for template rendering
        
    Returns:
        Rendered template as a string
    """
    template_content = load_template(template_name)
    
    if context:
        try:
            rendered_content = template_content.format(**context)
            return rendered_content
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {str(e)}")
            raise ValueError(f"Could not render template {template_name}: {str(e)}")
    
    return template_content
