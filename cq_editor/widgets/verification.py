"""Visible measured checks and an editable list of design requirements."""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QDialogButtonBox,
    QMessageBox,
    QHeaderView,
)

from ..ai.verification import KINDS, normalize_requirements


class VerificationPanel(QWidget):
    verify_requested = pyqtSignal()
    requirements_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.requirements = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.summary = QLabel("Design verification: not checked")
        self.summary.setWordWrap(True)
        bar.addWidget(self.summary, 1)
        self.checks_btn = QPushButton("Checks…")
        self.verify_btn = QPushButton("Verify")
        self.verify_btn.setToolTip(
            "Measure the current editor design against the listed checks"
        )
        bar.addWidget(self.checks_btn)
        bar.addWidget(self.verify_btn)
        layout.addLayout(bar)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Requirement", "Status", "Measured", "Expected"])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(100)
        self.tree.setMaximumHeight(170)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for index in (2, 3):
            self.tree.header().setSectionResizeMode(index, QHeaderView.Interactive)
            self.tree.setColumnWidth(index, 105)
        layout.addWidget(self.tree)
        self.verify_btn.clicked.connect(self.verify_requested)
        self.checks_btn.clicked.connect(self.edit_checks)

    def set_requirements(self, requirements):
        self.requirements = list(requirements)
        self.invalidate()

    def invalidate(self):
        self.tree.clear()
        self.summary.setText("Design verification: not checked")

    def set_report(self, report):
        self.tree.clear()
        counts = {}
        for result in report.get("rows", []):
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            item = QTreeWidgetItem(
                [
                    result["label"],
                    status,
                    result.get("measured", ""),
                    result.get("expected", ""),
                ]
            )
            for col in range(4):
                item.setToolTip(
                    col,
                    "\n".join(
                        value
                        for value in (
                            result["label"],
                            status,
                            result.get("measured", ""),
                            result.get("expected", ""),
                            result.get("detail", ""),
                        )
                        if value
                    ),
                )
            self.tree.addTopLevelItem(item)
        self.summary.setText(
            "Verification: "
            + ", ".join(
                f"{counts.get(s, 0)} {s.lower()}"
                for s in ("Passed", "Failed", "Not checked")
            )
        )

    def edit_checks(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Design requirements")
        dialog.resize(920, 420)
        layout = QVBoxLayout(dialog)
        help_text = QLabel(
            "Record the dimensions and features from your request. Dimensions and clearance are in mm; overlap is in mm³. Set Part to a displayed object name for a check about that part, or all for the whole model. Bore checks cover cylindrical holes only. Check tooltips show the scope used."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(
            [
                "Requirement",
                "Check",
                "Part",
                "Other part",
                "Axis",
                "Expected",
                "Tolerance",
                "Source text",
            ]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(table)

        def add(check=None):
            check = check or {}
            i = table.rowCount()
            table.insertRow(i)
            selector = QComboBox()
            for key, label in KINDS.items():
                selector.addItem(label, key)
            selector.setCurrentIndex(
                max(0, selector.findData(check.get("kind", "extent")))
            )
            table.setCellWidget(i, 1, selector)
            for col, key, default in [
                (0, "label", ""),
                (2, "target", ""),
                (3, "other", ""),
                (4, "axis", "X"),
                (5, "expected", ""),
                (6, "tolerance", "0.01"),
                (7, "source", ""),
            ]:
                table.setItem(i, col, QTableWidgetItem(str(check.get(key, default))))

        for check in self.requirements:
            add(check)
        controls = QHBoxLayout()
        add_btn, remove_btn = QPushButton("Add check"), QPushButton("Remove selected")
        add_btn.clicked.connect(lambda: add())
        remove_btn.clicked.connect(
            lambda: (
                table.removeRow(table.currentRow()) if table.currentRow() >= 0 else None
            )
        )
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addStretch()
        layout.addLayout(controls)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.rejected.connect(dialog.reject)

        def save():
            checks = []
            for i in range(table.rowCount()):
                check = {
                    key: table.item(i, col).text()
                    for col, key in [
                        (0, "label"),
                        (2, "target"),
                        (3, "other"),
                        (4, "axis"),
                        (5, "expected"),
                        (6, "tolerance"),
                        (7, "source"),
                    ]
                }
                check["kind"] = table.cellWidget(i, 1).currentData()
                checks.append(check)
            try:
                checks = normalize_requirements(checks)
            except (ValueError, TypeError) as error:
                QMessageBox.warning(dialog, "Check needs correction", str(error))
                return
            self.set_requirements(checks)
            self.requirements_changed.emit(checks)
            dialog.accept()

        buttons.accepted.connect(save)
        layout.addWidget(buttons)
        dialog.exec_()
