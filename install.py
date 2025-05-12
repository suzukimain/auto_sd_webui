import subprocess
import sys
import re

_deps = [
    "auto_diffusers>=2.0.22",
    "transformers<=4.32.1",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

def install_dependencies(deps):
    for package in deps.values():
        subprocess.run([sys.executable, "-m", "pip", "install", package])

install_dependencies(deps)

from auto_diffusers.pipeline_easy import search_civitai,search_huggingface
