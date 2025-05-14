import subprocess
import sys
import re

_deps = [
    "diffusers",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

def install_dependencies(deps):
    for package in deps.values():
        subprocess.run([sys.executable, "-m", "pip", "install", package])

install_dependencies(deps)