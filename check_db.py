from database import get_connection
import pandas as pd

conn = get_connection()

print("\n===== Houses =====")
print(pd.read_sql("SELECT * FROM houses LIMIT 10", conn))

print("\n===== Meter Readings =====")
print(pd.read_sql("SELECT * FROM meter_readings LIMIT 10", conn))

print("\n===== Total Houses =====")
print(pd.read_sql("SELECT COUNT(*) AS total FROM houses", conn))

print("\n===== Total Meter Readings =====")
print(pd.read_sql("SELECT COUNT(*) AS total FROM meter_readings", conn))

conn.close()