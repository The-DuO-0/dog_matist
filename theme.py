APP_QSS = r"""
QWidget {
    background: #111318;
    color: #e8ebf2;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog { background: #0c0e13; }
QFrame#Sidebar { background: #0a0c10; border-right: 1px solid #222632; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #f5f7fb; }
QLabel#Subtle { color: #8d96a8; }
QLabel#SectionTitle { font-size: 24px; font-weight: 700; }
QLabel#CardValue { font-size: 25px; font-weight: 700; }
QFrame#Card { background: #171a21; border: 1px solid #282d38; border-radius: 14px; }
QFrame#Panel { background: #15181f; border: 1px solid #282d38; border-radius: 14px; }
QPushButton {
    background: #202530; border: 1px solid #323846; border-radius: 9px;
    padding: 9px 13px; color: #edf0f6; font-weight: 600;
}
QPushButton:hover { background: #292f3b; }
QPushButton:pressed { background: #181c24; }
QPushButton#Primary { background: #4169e1; border-color: #4169e1; color: white; }
QPushButton#Primary:hover { background: #4d74e8; }
QPushButton#Danger { background: #472126; border-color: #6d3038; color: #ffd7dc; }
QPushButton#Nav { background: transparent; border: 0; text-align: left; padding: 11px 14px; color: #aeb6c6; }
QPushButton#Nav:hover { background: #161922; color: #ffffff; }
QPushButton#Nav:checked { background: #1c2230; color: #ffffff; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QListWidget, QTableWidget {
    background: #0f1217; border: 1px solid #2b303b; border-radius: 8px; padding: 7px;
    selection-background-color: #385dc7;
}
QHeaderView::section { background: #181c24; color: #aeb6c6; padding: 7px; border: 0; }
QTabWidget::pane { border: 0; }
QProgressBar { background: #0e1116; border: 1px solid #2b303b; border-radius: 7px; text-align: center; }
QProgressBar::chunk { background: #4169e1; border-radius: 6px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #303642; border-radius: 5px; min-height: 28px; }
"""
