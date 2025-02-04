# qpcr_analysis
Code used for analysis of qPCR data in DOI: 10.1126/sciadv.adi5945

To reproduce the analysis (within sampling error), clone this repository,
set up the conda environment definec in `conda_environment.yaml`,
activate the environment,
and run:

```bash
# qpcr analysis
python analysis_partI.py
# organize results into table
python analysis_partII.py
```
