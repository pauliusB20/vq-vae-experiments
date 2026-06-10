CMS Data Compression experiment
---------

Repository contains VQ-VAE and VAE experiments based on
CMS pileup dataset
---------

File structure:
* notebooks - folder contains VQ-VAE and VAE notebooks related to the CMS data compression experiment
* scripts - folder containing helper scripts
* bin - built models that are ready for testing in software
* examples - old experiment example code
* results - folder containing experiment plots and analytical reports
* helpers - folder that contains all the required helper classes and functions

NOTES (2026-05-18):
-------------------
* notebooks/mvp_vq.ipynb - recent VQ-VAE experiment notebook with marginally perspective results
* notebooks/mvp.ipynb - recent VAE experiment notebook with marginally perspective results

Setup:
---------------------
Create vqvae python virtual environment
1) python3 -m venv vq_vae <br/>

Activate env <br/>
2) In notebooks, select created virtual environment <br/>

Install packages <br/>
3) uv pip3 install -r --no-cache-dir requirements.txt <br/>
