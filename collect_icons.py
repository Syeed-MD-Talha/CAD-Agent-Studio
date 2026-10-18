from glob import glob
from subprocess import call
from os import remove

TEMPLATE = """<RCC>
  <qresource prefix="/images">
    {}
  </qresource>
</RCC>"""

ITEM_TEMPLATE = "<file>{}</file>"

QRC_OUT = "icons.qrc"
RES_OUT = "cq_editor/icons_res.py"
TOOL = "pyrcc5"

items = []

for pattern in ("icons/*.svg", "icons/*.png"):
    for i in glob(pattern):
        items.append(ITEM_TEMPLATE.format(i))


qrc_text = TEMPLATE.format("\n".join(items))

with open(QRC_OUT, "w") as f:
    f.write(qrc_text)

call([TOOL, QRC_OUT, "-o", RES_OUT])
remove(QRC_OUT)
