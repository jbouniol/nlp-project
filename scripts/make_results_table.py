"""results/ -> reports/results_table.md (fails if a run misses one of its artefacts)."""

from ledgar import runs
from ledgar.config import ROOT


def main() -> None:
    table = runs.results_table(ROOT / "results")
    markdown = runs.to_markdown(table)
    out = ROOT / "reports" / "results_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(markdown, end="")
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
