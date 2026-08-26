from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QWidget


def scroll_page(page: QWidget, *, min_content_height: int = 0) -> QScrollArea:
    """Host a dense Studio page in a vertical scroll area.

    Evolution and Dynasty intentionally contain a lot of durable telemetry. On a
    laptop-sized window Qt otherwise compresses those cards until labels/tables
    overlap. Giving the page a sensible content-height floor lets the outer page
    scroll naturally while keeping the existing inner table scrollbars.
    """

    if min_content_height < 0:
        raise ValueError("min_content_height must be non-negative")

    page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    if min_content_height:
        page.setMinimumHeight(int(min_content_height))

    scroll = QScrollArea()
    scroll.setObjectName("PageScroll")
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroll.setAutoFillBackground(False)
    scroll.viewport().setAutoFillBackground(False)
    scroll.setWidget(page)
    return scroll
