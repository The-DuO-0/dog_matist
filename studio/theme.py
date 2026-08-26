APP_QSS = r"""
QWidget {
    background: #111318;
    color: #e8ebf2;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog { background: #0c0e13; }
QFrame#Sidebar { background: #0a0c10; border-right: 1px solid #222632; }
QLabel#Brand { font-size: 23px; font-weight: 800; color: #f5f7fb; }
QLabel#Subtle { color: #8d96a8; }
QLabel#SectionTitle { font-size: 24px; font-weight: 700; }
QLabel#CardValue { font-size: 25px; font-weight: 700; }
QLabel#InfoNote {
    color: #bac8ee; background: #151d31; border: 1px solid #273c70;
    border-radius: 9px; padding: 10px 12px;
}
QLabel#RunBadge, QLabel#SidebarStatus {
    color: #aeb6c6; background: #181c24; border: 1px solid #2b303b;
    border-radius: 10px; padding: 6px 10px; font-weight: 700;
}
QLabel#RunBadge[active="true"], QLabel#SidebarStatus[active="true"] {
    color: #dce6ff; background: #17254b; border-color: #4169e1;
}
QLabel#StageBox {
    color: #939caf; background: #11151c; border: 1px solid #2a303b;
    border-radius: 10px; padding: 8px; font-weight: 700;
}
QLabel#StageBox[active="true"] {
    color: #ffffff; background: #1b2b59; border: 2px solid #527cff;
}
QFrame#Card { background: #171a21; border: 1px solid #282d38; border-radius: 14px; }
QFrame#Panel { background: #15181f; border: 1px solid #282d38; border-radius: 14px; }
QPushButton {
    background: #202530; border: 1px solid #323846; border-radius: 9px;
    padding: 9px 13px; color: #edf0f6; font-weight: 600;
}
QPushButton:hover {
    background: #2b3240; border-color: #59657a; color: #ffffff;
}
QPushButton:pressed {
    background: #151922; border-color: #7793e8; padding-top: 10px; padding-bottom: 8px;
}
QPushButton:disabled {
    background: #171a20; border-color: #252a33; color: #606878;
}
QPushButton#Primary { background: #4169e1; border-color: #4169e1; color: white; }
QPushButton#Primary:hover { background: #5278ec; border-color: #7695f3; }
QPushButton#Primary:pressed { background: #3157c6; border-color: #91a8ef; }
QPushButton#Danger { background: #472126; border-color: #6d3038; color: #ffd7dc; }
QPushButton#Danger:hover { background: #642b33; border-color: #a94e5a; }
QPushButton#Danger:pressed { background: #35191d; }
QPushButton#Nav {
    background: transparent; border: 1px solid transparent; text-align: left;
    padding: 11px 14px; color: #aeb6c6;
}
QPushButton#Nav:hover { background: #161d2b; border-color: #26334f; color: #ffffff; }
QPushButton#Nav:pressed { background: #10151f; }
QPushButton#Nav:checked { background: #1c2740; border-color: #304978; color: #ffffff; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget {
    background: #0f1217; border: 1px solid #2b303b; border-radius: 8px; padding: 7px;
    selection-background-color: #385dc7;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #53617a;
}
QHeaderView::section { background: #181c24; color: #aeb6c6; padding: 7px; border: 0; }
QTabWidget::pane { border: 0; }
QProgressBar {
    min-height: 14px; background: #0e1116; border: 1px solid #2b303b;
    border-radius: 7px; text-align: center;
}
QProgressBar::chunk { background: #4169e1; border-radius: 6px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #303642; border-radius: 5px; min-height: 28px; }
QToolTip { background: #202530; color: #ffffff; border: 1px solid #49536a; padding: 5px; }
"""
