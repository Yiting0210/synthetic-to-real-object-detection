import sys, re, nbformat
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
parts = re.split(r"^# %%(.*)$", text, flags=re.M)
cells = []
for head, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in head:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
nbformat.write(nb, dst)
print(len(cells), "cells ->", dst)
