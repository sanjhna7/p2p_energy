from database import get_connection
import pandas as pd

conn = get_connection()

print("\n===== Grid State =====")
print(pd.read_sql("SELECT * FROM grid_state", conn))

print("\n===== Transitions =====")
print(pd.read_sql("SELECT * FROM transitions", conn))

conn.close()