"""
Parser for Cython-generated HTML annotation files.
"""

from bs4 import BeautifulSoup
import logging
import os

logger = logging.getLogger("cython_analyzer.html")


def parse_annotation_html(html_content):
    """
    Parse the Cython-generated HTML annotation to extract optimization metrics.

    Args:
        html_content: HTML content from Cython annotation

    Returns:
        dict: Metrics including Python interaction, C operations, etc.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Initialize metrics
        metrics = {
            "python_lines": 0,
            "c_lines": 0,
            "py_object_interactions": 0,
            "gil_operations": 0,
            "vectorizable_loops": 0,
            "unoptimized_math": 0,
            "line_categories": {},  # Will store line number -> category mappings
        }

        # Process each line
        processed_lines = 0
        for line in soup.find_all("div", class_="line"):
            try:
                line_num = int(line.get("id", "0").replace("line", ""))
                processed_lines += 1

                # Skip empty lines
                if not line.text.strip():
                    continue

                # Categories based on Cython's color coding
                category = None

                # Check for Python interaction (yellow background)
                if line.find(
                    "span", style=lambda s: s and "background-color: #FFFF00" in s
                ):
                    metrics["python_lines"] += 1
                    metrics["py_object_interactions"] += 1
                    category = "python_interaction"

                # Check for GIL acquisition (pink background)
                elif line.find(
                    "span", style=lambda s: s and "background-color: #FFABAB" in s
                ):
                    metrics["gil_operations"] += 1
                    category = "gil_acquisition"

                # Check for pure C operations (no special highlighting)
                elif not line.find(
                    "span", style=lambda s: s and "background-color:" in s
                ):
                    metrics["c_lines"] += 1
                    category = "c_operation"

                # Check for loops that could be vectorized
                if (
                    "for" in line.text
                    and "range" in line.text
                    and not "prange" in line.text
                ):
                    metrics["vectorizable_loops"] += 1
                    category = category or "vectorizable_loop"

                # Check for unoptimized math operations
                if (
                    any(op in line.text for op in ["+", "-", "*", "/"])
                    and "double" in line.text
                ):
                    if not line.find(
                        "span", style=lambda s: s and "color: #0000FF" in s
                    ):
                        metrics["unoptimized_math"] += 1
                        category = category or "unoptimized_math"

                # Store the category for this line
                if category:
                    metrics["line_categories"][line_num] = category
            except Exception as e:
                logger.warning(f"Error processing line in HTML: {str(e)}")

        # If we didn't parse any lines from the HTML but we have static metrics,
        # use the static metrics to estimate
        if processed_lines == 0:
            logger.warning(
                "No lines processed from HTML annotation, using estimates from static analysis"
            )
            # We'll calculate these in the main analyze_code method
            pass
        else:
            metrics["total_lines"] = processed_lines

        logger.info(f"Parsed annotation HTML, processed {processed_lines} lines")
        return metrics

    except Exception as e:
        logger.error(f"Error parsing HTML annotation: {str(e)}")
        return {"error": f"Error parsing HTML annotation: {str(e)}"}


def locate_html_file(directory, module_name, pyx_path):
    """
    Locate the Cython-generated HTML annotation file.

    Args:
        directory: Directory to search in
        module_name: Module name
        pyx_path: Path to the original .pyx file

    Returns:
        str or None: Path to the HTML file if found, None otherwise
    """
    import os

    # Possible locations for HTML files
    possible_paths = [
        os.path.join(directory, f"{module_name}.html"),  # module_name.html
        f"{pyx_path}.html",  # file.pyx.html
        os.path.join(
            os.path.dirname(pyx_path), f"{module_name}.html"
        ),  # dir/module_name.html
    ]

    # Check each possible location
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"HTML annotation found at {path}")
            return path

    # Check for any HTML files in the directory
    if os.path.exists(directory):
        html_files = [f for f in os.listdir(directory) if f.endswith(".html")]
        if html_files:
            logger.info(f"Found HTML files in directory: {html_files}")
            return os.path.join(directory, html_files[0])

    logger.error("HTML annotation not found in any expected location")
    return None
