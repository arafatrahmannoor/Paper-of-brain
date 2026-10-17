"""Convert paper/main.tex into an editable Word file (paper/main.docx).

Usage: python tools/latex_to_docx.py BUILD_DIR CSL_FILE

Run after a full pdflatex/bibtex build, because figure, table, section and
equation numbers are taken from paper/main.aux so that the Word file is numbered
exactly like the PDF. BUILD_DIR must already contain figures/*.png: the PNG figures from paper/figures
plus the PDF figures and the TikZ architecture rendered at 300 dpi (pdftoppm -r 300).
CSL_FILE is an IEEE citation style, e.g. ieee.csl from the citeproc-py-styles package.
Styles (Times New Roman, A4) come from tools/word_reference.docx.
"""
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
build, csl = Path(sys.argv[1]), sys.argv[2]

tex = (PAPER / "main.tex").read_text()
aux = (PAPER / "main.aux").read_text()
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{(\\mbox\s*\{)?([^}]*)\}", aux):
    labels[m.group(1)] = m.group(3)  # section numbers are stored as \mbox {IV-B}

title = re.search(r"\\title\{(.+?)\}\n", tex).group(1)
body = tex[tex.index(r"\begin{document}") + len(r"\begin{document}"):tex.index(r"\end{document}")]

# Front matter handled by pandoc metadata; abstract and keywords kept as text.
body = re.sub(r"\\title\{.*?\}\n", "", body)
body = body[body.index(r"\maketitle") + len(r"\maketitle"):]
body = body.replace(r"\begin{IEEEkeywords}", r"\noindent\textit{Index Terms}---").replace(r"\end{IEEEkeywords}", "")
body = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", body)

# Custom macros.
body = re.sub(r"\\figorplaceholder\{([^}]+)\}\{[^}]*\}\{[^}]*\}", r"\\includegraphics{\1}", body)
body = body.replace(r"\input{figures/fig_architecture.tex}", r"\includegraphics{fig_architecture.png}")
body = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}.]+)\.pdf\}", r"\\includegraphics{\1.png}", body)
FIG_WIDTH = {"fig_confusion.png": "3.6in", "fig4_reliability.png": "3.6in", "fig_architecture.png": "3.4in",
             "fig_cm_models.png": "5.6in"}  # everything else spans the 6.27in A4 text width
body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}",
              lambda m: f"\\includegraphics[width={FIG_WIDTH.get(m.group(2), '6.2in')}]{{figures/{m.group(2)}}}", body)
body = re.sub(r"\\fname\{", r"\\texttt{", body)
body = re.sub(r"\\todo\{([^}]*)\}", r"[\1]", body)
body = body.replace(r"\R}", r"\mathbb{R}}").replace(r"\R^", r"\mathbb{R}^")
body = re.sub(r"\\shortstack\{([^}]*)\}", lambda m: m.group(1).replace(r"\\", " "), body)
body = body.replace("figure*", "figure").replace("table*", "table")
body = body.replace(r">{\raggedright\arraybackslash}", "")  # pandoc drops the first cell token otherwise
# Inside tables, write "a \pm b" as plain text so cells can wrap and stay easy to edit.
def plain_pm(tab):
    t = re.sub(r"\$\\mathbf\{([0-9.]+) \\pm ([0-9.]+)\}\$", lambda m: f"\\textbf{{{m.group(1)} \u00b1 {m.group(2)}}}", tab.group(0))
    return re.sub(r"\$([0-9.]+) \\pm ([0-9.]+)\$", lambda m: f"{m.group(1)} \u00b1 {m.group(2)}", t)


body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", plain_pm, body, flags=re.S)
# A "<" inside math is mis-rendered by some Word viewers; write these p-values as text.
body = body.replace(r"$p < 10^{-200}$", "\\textit{p} < 10\\textsuperscript{−200}")
body = body.replace(r"$<10^{-200}$", "< 10\\textsuperscript{−200}")
body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "", body)


# \IfFileExists{file}{yes}{no} -> yes (all figure files exist).
def strip_iffileexists(s):
    key = r"\IfFileExists{"
    while key in s:
        i = s.index(key)
        j = s.index("}", i) + 1  # end of the file argument

        def group(k):
            assert s[k] == "{"
            depth = 0
            for p in range(k, len(s)):
                depth += {"{": 1, "}": -1}.get(s[p], 0)
                if depth == 0:
                    return p
        yes_end = group(j)
        no_end = group(yes_end + 1)
        s = s[:i] + s[j + 1:yes_end] + s[no_end + 1:]
    return s


body = strip_iffileexists(body)


# align -> gather without alignment marks (Word and LibreOffice both render this cleanly);
# a further & on a line separates side-by-side equations and becomes a \qquad.
def flatten_align(m):
    lines = []
    for line in m.group(1).split("\\\\"):
        parts = line.split("&")
        pairs = ["".join(parts[k:k + 2]) for k in range(0, len(parts), 2)]
        lines.append(r" \qquad ".join(p_ for p_ in pairs if p_.strip()))
    return "\\begin{gather}" + "\\\\".join(lines) + "\\end{gather}"


body = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", flatten_align, body, flags=re.S)
body = body.replace(r"\nonumber", "")

# Number sections like IEEEtran (I, II ... / A, B ...); unnumbered starred sections stay plain.
roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
sec_n, sub_n = 0, 0


def number_heading(m):
    global sec_n, sub_n
    cmd, star, text = m.group(1), m.group(2), m.group(3)
    if star:
        return m.group(0)
    if cmd == "section":
        sec_n, sub_n = sec_n + 1, 0
        return f"\\section*{{{roman[sec_n - 1]}. {text}}}"
    sub_n += 1
    return f"\\subsection*{{{chr(64 + sub_n)}. {text}}}"


body = re.sub(r"\\(section|subsection)(\*?)\{([^{}]*)\}", number_heading, body)

# Caption prefixes from the labels inside each float.
def prefix_caption(m):
    env, content = m.group(1), m.group(2)
    lab = re.search(r"\\label\{([^}]+)\}", content)
    if lab and lab.group(1) in labels:
        tag = ("Fig. " if env == "figure" else "TABLE ") + labels[lab.group(1)] + ". "
        content = content.replace(r"\caption{", r"\caption{" + tag, 1)
    return f"\\begin{{{env}}}{content}\\end{{{env}}}"


body = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", prefix_caption, body, flags=re.S)

# Cross-references -> literal numbers.
body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "??"), body)
body = re.sub(r"\\label\{[^}]+\}", "", body)

meta = build / "meta.yaml"
meta.write_text(
    "---\n"
    f"title: \"{title}\"\n"
    "author:\n"
    "  - Arafat Rahman\n"
    "  - Md Assaduzzaman\n"
    "date: \"Department of Computer Science and Engineering, Daffodil International University, Dhaka, "
    "Bangladesh. Corresponding author: Arafat Rahman (arafat2001noor@gmail.com)\"\n"
    "reference-section-title: References\n"
    "link-citations: true\n"
    "---\n"
)
src = build / "body.tex"
src.write_text("\\begin{document}\n" + body + "\n\\end{document}\n")
cmd = ["pandoc", str(src), "--metadata-file", str(meta), "-f", "latex", "-o", str(PAPER / "main.docx"),
       "--citeproc", "--bibliography", str(PAPER / "references.bib"), "--csl", csl,
       "--resource-path", str(build), "--reference-doc", str(Path(__file__).with_name("word_reference.docx")),
       "--dpi", "300"]
subprocess.run(cmd, check=True, cwd=build)


# Full-width tables, columns sized by content (pandoc gives every column the same narrow width).
def widen_tables(docx_path, text_width=9026):  # A4 with 1-inch margins
    import zipfile
    zin = zipfile.ZipFile(docx_path)
    files = {n: zin.read(n) for n in zin.namelist()}
    zin.close()
    xml = files["word/document.xml"].decode("utf8")

    def fix(tbl):
        t = tbl.group(0)
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", t, flags=re.S)
        ncol = len(re.findall(r"<w:gridCol ", t))
        need = [4] * ncol
        for r in rows:
            cells = re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S)
            if len(cells) != ncol:
                continue
            for k, c in enumerate(cells):
                txt = "".join(re.findall(r"<(?:w|m):t[^>]*>([^<]*)</(?:w|m):t>", c))
                # equations cannot wrap, so they need their full length; plain text can wrap
                longest = max((len(w_) for w_ in txt.split()), default=0) + 1
                n = min(len(txt) + 1, 24) if "<m:oMath" in c else max(longest + 2, min(len(txt), 30) * 0.45)
                need[k] = max(need[k], n)
        weights = [max(n, 6) for n in need]
        widths = [int(text_width * w / sum(weights)) for w in weights]
        widths[-1] += text_width - sum(widths)
        grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}" />' for w in widths) + "</w:tblGrid>"
        t = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", grid, t, flags=re.S)
        t = t.replace('<w:tblW w:type="auto" w:w="0" />', f'<w:tblW w:type="dxa" w:w="{text_width}" />')

        def cellw(r):
            row = r.group(0)
            cells = re.findall(r"<w:tc>.*?</w:tc>", row, flags=re.S)
            if len(cells) != ncol:
                return row
            for k, c in enumerate(cells):
                c2 = re.sub(r"<w:tcW [^>]*/>", "", c)
                c2 = c2.replace("<w:tcPr>", f'<w:tcPr><w:tcW w:w="{widths[k]}" w:type="dxa" />', 1) if "<w:tcPr>" in c2 \
                    else c2.replace("<w:tc>", f'<w:tc><w:tcPr><w:tcW w:w="{widths[k]}" w:type="dxa" /></w:tcPr>', 1)
                row = row.replace(c, c2, 1)
            return row
        t = re.sub(r"<w:tr[ >].*?</w:tr>", cellw, t, flags=re.S)
        # 9-pt table text, as usual in journals (text runs and equation runs)
        sz = '<w:sz w:val="18" /><w:szCs w:val="18" />'
        t = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr>" + sz + "</w:rPr>", t)
        t = t.replace("<w:r><w:rPr>", "<w:r><w:rPr>" + sz) if False else re.sub(r"<w:r><w:rPr>(?!<w:sz)", "<w:r><w:rPr>" + sz, t)
        t = re.sub(r"<m:r>(<m:rPr>.*?</m:rPr>)?", lambda m: "<m:r>" + (m.group(1) or "") + "<w:rPr>" + sz + "</w:rPr>", t)
        return t

    xml = re.sub(r"<w:tbl>.*?</w:tbl>", fix, xml, flags=re.S)
    xml = re.sub(r"<w:sectPr\s*/>", "", xml)  # pandoc writes an empty one
    if "<w:sectPr" not in xml:
        xml = xml.replace("</w:body>", '<w:sectPr><w:pgSz w:w="11906" w:h="16838" /><w:pgMar w:top="1440" '
                          'w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" '
                          'w:gutter="0" /></w:sectPr></w:body>')
    files["word/document.xml"] = xml.encode("utf8")
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)


widen_tables(PAPER / "main.docx")
print("wrote", PAPER / "main.docx")
