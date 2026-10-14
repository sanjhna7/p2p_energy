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

## Prices

**[R8] AER, Default market offer prices 2026–27: Final determination, May 2026.**
Figure 2.1: Ausgrid residential flat-rate DMO tariff caps, usage
33.1372 c/kWh, supply 166.2289 c/day; "All tariff caps include GST".
<https://www.aer.gov.au/system/files/2026-05/AER%20-%20Final%20determination%20-%20Default%20market%20offer%202026%E2%80%9327.pdf>,
SHA-256 `8c062224…03dd2`. CC BY 4.0.

**[R9] IPART, Fact Sheet: Solar feed-in tariff benchmark for 2026-27, 25 May 2026.**
All-day benchmark 3.4 to 6.5 c/kWh, 1 Jul 2026 – 30 Jun 2027. GST not stated.
<https://www.ipart.nsw.gov.au/sites/default/files/cm9_documents/Fact-Sheet-All-day-solar-feed-in-tariff-benchmark-tariff-2026-27.PDF>,
SHA-256 `08c16056…51680`.

## Placeholders pending the EEE specification

**[R10] ETSI EN 300 132-3 V2.3.1 (2023-01), Power supply interface at the
input to ICT equipment; Part 3: Up to 400 V DC.** Clause 5.2: normal
service voltage range 260–400 V DC. Clause 5.3 note 1: 354 V and 380 V
as example normal operating voltages. Used only as a placeholder for the
community DC bus until the EEE team's values arrive.
<https://www.etsi.org/deliver/etsi_en/300100_300199/30013203/02.03.01_60/en_30013203v020301p.pdf>,
SHA-256 `41138fef…1ec0e`.

Full hashes are in `config/p2p.yaml` (R5, R6, R8–R10) and `data/README.md` (R1, R3).
Local copies of R5, R6, R8–R10 are kept in `data/raw/references/` (not committed).
