import sqlite3 as sq

con = None
c = None


def start_db_connection():
    global con
    global c
    con = sq.connect(f'D:/D2_Datamining/Package Unpacker/db/2_9_2_1_all.db')
    c = con.cursor()


def get_entries_from_table(pkg_str, column_select='*'):
    global c
    c.execute(f'SELECT {column_select} from {pkg_str}')
    rows = c.fetchall()
    return rows
