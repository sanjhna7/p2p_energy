# References

Every source the code depends on. Each entry was checked on 2026-10-14:
DOIs against Crossref, URLs by download, and file hashes recorded where
we hold a copy. Add nothing here that has not been checked the same way.

## Data

**[R1] Ausgrid Solar Home Electricity Data, 2010–2013.**
Ausgrid, "Solar home electricity data – notes (August 2014)", 22 Aug 2014.
300 gross-metered solar homes, half-hourly kWh, 1 Jul 2010 – 30 Jun 2013.
Released under Creative Commons Attribution 3.0 Australia.
The Ausgrid page is withdrawn; the metadata record remains at
<https://data.gov.au/data/en/dataset/nsw-solar-home-electricty-data>.
Archive copy used: <https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip>
(P. Haessig, May 2026), SHA-256 `5a766f52…5c5d82`.

**[R2] Paper describing R1.**
E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray,
"Residential load and rooftop PV generation: an Australian distribution
network dataset", *International Journal of Sustainable Energy*,
vol. 36, no. 8, pp. 787–806, 2017 (online 27 Oct 2015).
doi:[10.1080/14786451.2015.1100196](https://doi.org/10.1080/14786451.2015.1100196)

## Network

**[R3] IEEE European Low Voltage Test Feeder.**
IEEE PES Distribution System Analysis Subcommittee, Test Feeders Working
Group, "The IEEE European Low Voltage Test Feeder", documentation dated
19 May 2015. Radial, 50 Hz, 11/0.416 kV 0.8 MVA Dy transformer.
<https://cmte.ieee.org/pes-testfeeders/wp-content/uploads/sites/167/2017/08/European_LV_Test_Feeder_v2.zip>,
SHA-256 `47fcb818…46ddc`.

**[R4] Paper describing the IEEE test feeders.**
K. P. Schneider, B. A. Mather, B. C. Pal, C.-W. Ten, G. J. Shirek, H. Zhu,
J. C. Fuller, J. L. R. Pereira, L. F. Ochoa, L. R. de Araujo, R. C. Dugan,
S. Matthias, S. Paudyal, T. E. McDermott, W. Kersting,
"Analytic Considerations and Design Basis for the IEEE Distribution Test
Feeders", *IEEE Transactions on Power Systems*, vol. 33, no. 3,
pp. 3181–3188, May 2018.
doi:[10.1109/TPWRS.2017.2760011](https://doi.org/10.1109/TPWRS.2017.2760011)

## Equipment

**[R5] BYD Battery-Box Premium LVS datasheet, EN V1.0, 26 Jun 2024.**
<https://bydbatterybox.com/uploads/downloads/BBOX_LVS_Datasheet_EN_V1.0_240626_L-668f94bd814fd.pdf>,
SHA-256 `e786fba9…ef512`.
LVS 4.0: 4 kWh usable (100% DOD, 0.2C, 25 °C), 51.2 V nominal,
40–57.6 V operating, 65 A max continuous output, round-trip efficiency ≥ 95%.

**[R6] BYD Battery-Box Premium LVS datasheet, AU V2.2, 4 Jan 2023.**
<https://bydbatterybox.com/uploads/downloads/230318_BYD%20Battery-Box%20Premium%20LVS%20Datasheet_AU%20V2.2-64153555191fb.pdf>,
SHA-256 `c8af864c…b5b9f`.
Same figures as R5, plus rated DC power 3.3 kW for the LVS 4.0
(R5 does not list rated power).

## Standards

**[R7] AS IEC 60038, Standard voltages (Standards Australia).**
Sets the Australian nominal low-voltage supply at 230/400 V.
The standard is paywalled and was not read directly; the nominal value
was confirmed via a secondary source, PowerLogic, "LV Voltage Range and
Implications", <https://www.powerlogic.com.au/lv-voltage-range-and-implications/>
(checked 2026-10-14). Cite the standard itself in formal reports.

Full hashes are in `config/p2p.yaml` (R5, R6) and `data/README.md` (R1, R3).
