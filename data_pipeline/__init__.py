"""Data ingestion for the GAT-MARL P2P project.

ausgrid - real half-hourly household load + rooftop PV (Ausgrid Solar Home)
feeder  - IEEE European LV Test Feeder topology and line parameters
build   - joins the two into the processed time-series + graph files

See docs/dataset_decision.md for why these sources were chosen.
"""
