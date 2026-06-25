from passlib.context import CryptContext
import sqlite3

pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")
hashed = pwd_context.hash("Password123!")

conn = sqlite3.connect('backend/data/sol.db')
cursor = conn.cursor()
cursor.execute('UPDATE users SET hashed_password = ? WHERE email = ?', (hashed, "socilaw715@luxudata.com"))
conn.commit()
conn.close()
print("Password updated successfully!")
