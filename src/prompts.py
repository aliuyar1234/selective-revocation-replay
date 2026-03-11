from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptLibrary:
    def __init__(self, prompt_dir: str | Path):
        self.prompt_dir = Path(prompt_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **kwargs: object) -> str:
        return self.env.get_template(template_name).render(**kwargs)
