from database import initialize_database
from ingest import ingest_data
from state import get_state
from simulation import simulate_step


def main():

    print("=" * 50)
    print("P2P Energy Management System")
    print("=" * 50)

    print("\n1. Initializing database...")
    initialize_database()

    print("Database ready.")

    print("\n2. Importing CSV...")
    ingest_data()

    print("\n3. Reading current state...")

    timestamp = "2010-07-15 13:00:00"

    state = get_state(timestamp)

    print(f"Timestamp : {state['timestamp']}")
    print(f"Total Houses : {len(state['houses'])}")

    print("\nFirst House:")

    print(state["houses"][0])

    print("\n4. Running simulation...")

    simulate_step(timestamp)

    print("\n5. Reading UPDATED state...")
    updated_state = get_state(timestamp)
    print(updated_state["houses"][0]) 

    print("\nDone.")


if __name__ == "__main__":
    main()