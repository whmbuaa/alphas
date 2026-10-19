"""把因子 Markdown 打成同目录下的 PDF。图片按文档所在目录解析。"""
import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import markdown

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
CSS = """
@page { size: A4; margin: 14mm 12mm; }
body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 11pt; line-height: 1.45; color: #222; }
h1 { font-size: 18pt; }
h2 { font-size: 14pt; margin-top: 1.2em; }
h3 { font-size: 12pt; }
img { max-width: 100%; height: auto; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 10pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 6px; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
pre { white-space: pre-wrap; }
"""


def to_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def to_pdf(md_path: Path) -> Path:
    md_path = md_path.resolve()
    pdf_path = md_path.with_suffix(".pdf")
    html = to_html(md_path)
    local_html = md_path.with_suffix(".html")
    local_html.write_text(html, encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--allow-file-access-from-files",
                f"--user-data-dir={tmp}",
                f"--print-to-pdf={pdf_path}",
                local_html.as_uri(),
            ]
            subprocess.run(cmd, check=True)
    finally:
        local_html.unlink(missing_ok=True)
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError(f"PDF 没有写出来: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    files = []
    for root in args.roots:
        files.extend(sorted(root.rglob("README.md")))
    total = len(files)
    print(f"pdf {total} files, workers={args.workers}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(to_pdf, path): path for path in files}
        for fut in as_completed(futures):
            done += 1
            path = futures[fut]
            try:
                pdf = fut.result()
                print(f"[{done}/{total}] {pdf}", flush=True)
            except Exception as exc:
                print(f"[{done}/{total}] FAIL {path} {exc}", flush=True)


if __name__ == "__main__":
    main()
