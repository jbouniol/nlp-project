"""The test split is read only to predict and save, in a notebook cell or a script marked so (S01 p12)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT_READS = ('ledgar["test"]', "ledgar['test']")


def test_package_never_reads_the_test_split():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text()
        assert not any(s in text for s in TEST_SPLIT_READS), f"{path.name} reads the test split"


def test_scripts_read_the_test_split_only_to_predict_and_save():
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text()
        if any(s in text for s in TEST_SPLIT_READS):
            assert "predict & save" in text, f"{path.name} reads the test split outside a predict & save step"


def test_notebooks_read_the_test_split_only_to_predict_and_save():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        for i, cell in enumerate(json.loads(path.read_text())["cells"]):
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and any(s in source for s in TEST_SPLIT_READS):
                assert "predict & save" in source, f"{path.name} cell {i} reads the test split"
