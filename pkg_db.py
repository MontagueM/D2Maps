import sqlite3 as sq

con = None
c = None


def start_db_connection():
    global con
    global c
    version = '3_0_0_1'
    con = sq.connect(f'I:/d2_pkg_db/{version}.db')
    c = con.cursor()


def get_entries_from_table(pkg_str, column_select='*'):
    global c
    c.execute(f'SELECT {column_select} from {pkg_str}')
    rows = c.fetchall()
    return rows
