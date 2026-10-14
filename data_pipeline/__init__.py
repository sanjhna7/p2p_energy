"""Data ingestion for the GAT-MARL P2P project.

ausgrid      - real half-hourly household load + rooftop PV (Ausgrid Solar Home)
eulv_network - IEEE European LV Test Feeder topology and line parameters
build_mvp_dataset - joins the two into the MVP time-series + graph files

See docs/dataset_decision.md for why these sources were chosen.
"""
