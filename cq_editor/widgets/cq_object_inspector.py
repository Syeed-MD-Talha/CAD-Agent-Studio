from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QAction
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from OCP.AIS import AIS_ColoredShape
from OCP.gp import gp_Ax3

from build123d import Shape, Compound, Vector

from ..mixins import ComponentMixin
from ..icons import icon


class CQChildItem(QTreeWidgetItem):

    def __init__(self, cq_item, **kwargs):

        super(CQChildItem, self).__init__(
            [type(cq_item).__name__, str(cq_item)], **kwargs
        )

        self.cq_item = cq_item


class CQStackItem(QTreeWidgetItem):

    def __init__(self, name, workplane=None, **kwargs):

        super(CQStackItem, self).__init__([name, ""], **kwargs)

        self.workplane = workplane


class CQObjectInspector(QTreeWidget, ComponentMixin):

    name = "CQ Object Inspector"

    sigRemoveObjects = pyqtSignal(list)
    sigDisplayObjects = pyqtSignal(list, bool)
    sigShowPlane = pyqtSignal([bool], [bool, float])
    sigChangePlane = pyqtSignal(gp_Ax3)

    def __init__(self, parent):

        super(CQObjectInspector, self).__init__(parent)
        self.setHeaderHidden(False)
        self.setRootIsDecorated(True)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.setColumnCount(2)
        self.setHeaderLabels(["Type", "Value"])

        self.root = self.invisibleRootItem()
        self.inspected_items = []

        self._toolbar_actions = [
            QAction(
                icon("inspect"),
                "Inspect CQ object",
                self,
                toggled=self.inspect,
                checkable=True,
            )
        ]

        self.addActions(self._toolbar_actions)

    def menuActions(self):

        return {"Tools": self._toolbar_actions}

    def toolbarActions(self):

        return self._toolbar_actions

    @pyqtSlot(bool)
    def inspect(self, value):

        if value:
            self.itemSelectionChanged.connect(self.handleSelection)
            self.itemSelectionChanged.emit()
        else:
            self.itemSelectionChanged.disconnect(self.handleSelection)
            self.sigRemoveObjects.emit(self.inspected_items)
            self.sigShowPlane.emit(False)

    @pyqtSlot()
    def handleSelection(self):

        inspected_items = self.inspected_items
        self.sigRemoveObjects.emit(inspected_items)
        inspected_items.clear()

        items = self.selectedItems()
        if len(items) == 0:
            return

        item = items[-1]
        if type(item) is CQStackItem and getattr(item, "workplane", None) is not None:
            # Legacy Workplane-stack branch (CadQuery). build123d shapes
            # never create these, but keep the handler for compatibility.
            cq_plane = item.workplane.plane
            dim = item.workplane.largestDimension()
            plane = gp_Ax3(
                cq_plane.origin.toPnt(), cq_plane.zDir.toDir(), cq_plane.xDir.toDir()
            )
            self.sigChangePlane.emit(plane)
            self.sigShowPlane[bool, float].emit(True, dim)

            for child in (item.child(i) for i in range(item.childCount())):
                obj = child.cq_item
                if isinstance(obj, Shape) and not isinstance(obj, Vector):
                    ais = AIS_ColoredShape(obj.wrapped)
                    inspected_items.append(ais)

        else:
            self.sigShowPlane.emit(False)
            obj = getattr(item, "cq_item", None)
            if isinstance(obj, Shape) and not isinstance(obj, Vector):
                try:
                    ais = AIS_ColoredShape(obj.wrapped)
                    inspected_items.append(ais)
                except Exception:
                    pass

        self.sigDisplayObjects.emit(inspected_items, False)

    @pyqtSlot(object)
    def setObject(self, cq_obj):

        self.root.takeChildren()

        if not isinstance(cq_obj, Shape):
            return

        # Top item: the shape itself with label + key metrics.
        label = getattr(cq_obj, "label", "") or ""
        title = f"{type(cq_obj).__name__}" + (f" ({label})" if label else "")
        top = CQStackItem(title)
        # CQStackItem without workplane falls into the Shape branch of
        # handleSelection; stash the shape so selecting the top works.
        top.cq_item = cq_obj
        self.root.addChild(top)

        try:
            kind_groups = (
                ("Solids", cq_obj.solids()),
                ("Faces", cq_obj.faces()),
                ("Wires", cq_obj.wires()),
                ("Edges", cq_obj.edges()),
                ("Vertices", cq_obj.vertices()),
            )
        except Exception:
            kind_groups = ()

        for kind, members in kind_groups:
            try:
                members = list(members)
            except Exception:
                continue
            if not members:
                continue
            group = CQStackItem(f"{kind} ({len(members)})")
            group.cq_item = None
            top.addChild(group)
            for obj in members:
                group.addChild(CQChildItem(obj))

        # Assembly structure for Compounds with children.
        try:
            children = (
                tuple(cq_obj.children) if isinstance(cq_obj, Compound) else ()
            )
        except Exception:
            children = ()

        if children:
            agroup = CQStackItem(f"Children ({len(children)})")
            agroup.cq_item = None
            top.addChild(agroup)
            for child in children:
                clabel = getattr(child, "label", "") or type(child).__name__
                citem = CQChildItem(child)
                citem.setText(0, f"{type(child).__name__}")
                citem.setText(1, clabel)
                agroup.addChild(citem)

        top.setExpanded(True)
