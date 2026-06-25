import sqlite3
conn = sqlite3.connect('backend/data/sol.db')
cursor = conn.cursor()
cursor.execute('SELECT email, hashed_password, is_admin FROM users')
for row in cursor.fetchall():
    print(f"Email: {row[0]}, Hash: {row[1]}, Admin: {row[2]}")
conn.close()
