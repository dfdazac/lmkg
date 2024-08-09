import os

def get_template(name: str):
    """Get content of Jinja template from file"""
    templates_dir = os.path.dirname(__file__)
    template_path = os.path.join(templates_dir, 'template-files', f'{name}.jinja')
    with open(template_path, 'r') as f:
        return f.read()
