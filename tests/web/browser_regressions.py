"""Build a standalone offline UI test page. Open it in a browser to run the checks."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def build_test_page() -> str:
    root = Path(__file__).resolve().parents[2]
    app = (root / "web/index.html").read_text()
    app = re.sub(r"<link\b[^>]+>", "", app)
    app = app.replace("</head>", "<style>" + (root / "web/styles.css").read_text() + "</style></head>")
    # The ordinary startup health check and every user action stay offline.
    script = """
    window.fetch = async () => new Response("{}", {status: 503});
    """
    script += (root / "web/app.js").read_text()
    script += Path(__file__).with_suffix(".js").read_text()
    script += """
    window.addEventListener("load", async () => {
      const output = parent.document.getElementById("result");
      try {
        const checks = await runBrowserRegressions();
        output.textContent = "PASS " + checks.length + "\\n" + checks.join("\\n");
      } catch (error) {
        output.textContent = "FAIL " + error.stack;
      }
    });
    """
    app = app.replace('<script src="./app.js"></script>', "<script>" + script + "</script>")
    return (
        '<!doctype html><meta charset="utf-8"><title>Offline UI regression results</title>'
        '<pre id="result">Running</pre><iframe width="390" height="1000" style="border:0" '
        f'srcdoc="{html.escape(app, quote=True)}"></iframe>'
    )


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(build_test_page())
