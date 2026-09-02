"""
Provisional DC-nanogrid dataset generator for MARL development.

The temporal PV generation and household consumption profiles in this
package are REAL, taken from the Ausgrid Solar Home Electricity Dataset
(2010-2011). Every electrical variable the source dataset does not
contain - voltages, currents, battery state of charge, bus-tie power -
is produced by a simplified, physically consistent nanogrid model so
that the MARL environment can be built and tested before the
Simulink/hardware simulation is finished.

See docs/PROVISIONAL_DATASET.md before using any of these variables in
a publication.
"""
