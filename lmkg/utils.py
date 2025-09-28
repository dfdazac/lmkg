import logging
import os

import jinja2


def get_chat_template(name: str):
    """Get content of Jinja template from file"""
    templates_dir = os.path.dirname(__file__)
    template_path = os.path.join(templates_dir, 'templates', f'{name}.jinja')
    with open(template_path, 'r') as f:
        return f.read()


def build_task_input(task: str, task_kwargs: dict):
    env = jinja2.Environment(loader=jinja2.PackageLoader("lmkg",
                                                         "prompts"))
    prompt = env.get_template(f"{task}.jinja")
    return prompt.render(**task_kwargs)


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger('lmkg')
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger
