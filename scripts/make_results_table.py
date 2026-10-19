"""results/ -> reports/results_table.md, results/retrieval/ -> reports/retrieval_table.md."""

from ledgar import retrieval, runs
from ledgar.config import ROOT


def main() -> None:
    tables = {"results_table.md": runs.results_table(ROOT / "results")}
    if (ROOT / "results" / "retrieval").exists():
        tables["retrieval_table.md"] = retrieval.retrieval_table(ROOT / "results" / "retrieval")
    for name, table in tables.items():
        markdown = runs.to_markdown(table)
        out = ROOT / "reports" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print(markdown, end="")
        print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
