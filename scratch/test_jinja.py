import sys
from jinja2 import Environment, FileSystemLoader

try:
    env = Environment(loader=FileSystemLoader('frontend/templates'))
    # Jinja2 requires finding extending templates in the same environment
    env.get_template('app/automl_studio.html')
    print("SUCCESS: Template compiled successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
