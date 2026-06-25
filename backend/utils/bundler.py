import os
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger("SOL.DataBundler")
logger.setLevel(logging.INFO)

if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def create_studio_bundle(output_zip_path: str) -> str:
    """
    Dynamically compresses required Data Cleaning Studio directories (core/, utils/, 
    backend/tools/audit/, backend/tools/viz_engine/) into a single studio_core.zip archive.
    
    Excludes python bytecode (__pycache__/, *.pyc), system dotfiles, and temporary outputs.
    
    Args:
        output_zip_path (str): Filepath where the ZIP bundle should be saved.
        
    Returns:
        str: Absolute path to the created ZIP archive.
    """
    # Resolve the project root dynamically (backend/utils/bundler.py -> backend/utils/ -> backend/ -> root/)
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = Path(output_zip_path)
    
    # Ensure destination folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Target directories to package
    targets = [
        "core",
        "utils",
        "backend/tools/audit",
        "backend/tools/viz_engine",
        "backend/tools/synthetic_data",
    ]
    
    logger.info("Initializing Data Studio source-code packaging from root: '%s'...", project_root)
    logger.info("Saving bundle to target destination: '%s'...", output_path)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for target in targets:
            target_dir = project_root / target
            if not target_dir.exists():
                logger.warning("Target directory '%s' does not exist. Skipping.", target_dir)
                continue
            
            logger.info("Zipping target directory: '%s'...", target)
            for root, dirs, files in os.walk(target_dir):
                # Exclude runtime caches to minimize archive size
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__")
                if ".git" in dirs:
                    dirs.remove(".git")
                
                for file in files:
                    # Ignore bytecode, hidden/system files, and temp logs
                    if file.endswith(('.pyc', '.pyo', '.pyd', '.zip')) or file.startswith('.'):
                        continue
                    
                    file_path = Path(root) / file
                    # Calculate path relative to project root to preserve relative imports inside runtime env
                    archive_name = file_path.relative_to(project_root)
                    
                    zipf.write(file_path, archive_name)
                    
    logger.info("Source-code packaging completed. Compressed file path: '%s'", output_path.resolve())
    return str(output_path.resolve())
