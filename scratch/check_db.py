import sqlite3

conn = sqlite3.connect('backend/data/sol.db')
cur = conn.cursor()
try:
    cur.execute('SELECT email, status, password_hash FROM users')
    print("Users:", cur.fetchall())
except Exception as e:
    print("Error:", e)
finally:
    conn.close()
