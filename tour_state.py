import sqlite3

conn = sqlite3.connect("customers.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        chat_id TEXT PRIMARY KEY,
        last_candidates TEXT
    )
""")

def save_conversation(chat_id, candidates):
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO customers VALUES (?, ?)",
        (chat_id, ",".join(candidates))
    )
    conn.commit()
    conn.close()

def get_customer_context(chat_id):
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM customers WHERE chat_id = ?",
        (chat_id,)
    )
    i = cursor.fetchone()
    conn.close()
    return i

def fetch_id_exists(chat_id):
    try: 
        conn = sqlite3.connect("customers.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM customers WHERE chat_id = ? LIMIT 1", (chat_id,))
        result = cursor.fetchone()
        return bool(result)
    
    except Exception as e:
        print(f"Database error: {e}")
        return False


conn.commit()
conn.close()

save_conversation("55555", ["Da Nang"])
print(fetch_id_exists("55555"))
print(fetch_id_exists("99999"))