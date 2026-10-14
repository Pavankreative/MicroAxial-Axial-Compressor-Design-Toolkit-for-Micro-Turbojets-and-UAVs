import json
import os
from microaxial.cli.state import MicroAxialDesign

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    return MicroAxialDesign.from_file(f"{TEST_DIR}/designs/{name}.json").definition


base_design = _load("base_design")
mark1 = _load("mark1")

micro_default = _load("micro_turbojet_default")
