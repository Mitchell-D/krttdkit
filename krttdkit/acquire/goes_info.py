"""
Description of ABI products from the NOAA github page here:
https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16

Also, hard-coded instances of valid ABI_Products (as of 20230817).
This list may need to be updated at times, which can be done by copying
the dictionary returned by GetGOES()._get_product_api().
"""

goes_descriptions = {
    "ABI-L1b-RadF": \
            "ABI Level 1b Full Disk",
    "ABI-L1b-RadC": \
            "ABI Level 1b CONUS",
    "ABI-L1b-RadM": \
            "ABI Level 1b Mesoscale",
    "ABI-L2-ACHAC": \
            "ABI Level 2 Cloud Top Height CONUS",
    "ABI-L2-ACHAF": \
            "ABI Level 2 Cloud Top Height Full Disk",
    "ABI-L2-ACHAM": \
            "ABI Level 2 Cloud Top Height Mesoscale",
    "ABI-L2-ACHTF": \
            "ABI Level 2 Cloud Top Temperature Full Disk",
    "ABI-L2-ACHTM": \
            "ABI Level 2 Cloud Top Temperature Mesoscale",
    "ABI-L2-ACMC": \
            "ABI Level 2 Clear Sky Mask CONUS",
    "ABI-L2-ACMF": \
            "ABI Level 2 Clear Sky Mask Full Disk",
    "ABI-L2-ACMM": \
            "ABI Level 2 Clear Sky Mask Mesoscale",
    "ABI-L2-ACTPC": \
            "ABI Level 2 Cloud Top Phase CONUS",
    "ABI-L2-ACTPF": \
            "ABI Level 2 Cloud Top Phase Full Disk",
    "ABI-L2-ACTPM": \
            "ABI Level 2 Cloud Top Phase Mesoscale",
    "ABI-L2-ADPC": \
            "ABI Level 2 Aerosol Detection CONUS",
    "ABI-L2-ADPF": \
            "ABI Level 2 Aerosol Detection Full Disk",
    "ABI-L2-ADPM": \
            "ABI Level 2 Aerosol Detection Mesoscale",
    "ABI-L2-AICEF": \
            "Ice Concentration and Extent",
    "ABI-L2-AITAF": \
            "Ice Age and Thickness",
    "ABI-L2-AODC": \
            "ABI Level 2 Aerosol Optical Depth CONUS",
    "ABI-L2-AODF": \
            "ABI Level 2 Aerosol Optical Depth Full Disk",
    "ABI-L2-BRFC": \
            "Land Surface Bidirectional Reflectance Factor (CONUS) 2 km resolution & DQFs",
    "ABI-L2-BRFF": \
            "Land Surface Bidirectional Reflectance Factor (Full Disk) 2 km resolution & DQFs",
    "ABI-L2-BRFM": \
            "Land Surface Bidirectional Reflectance Factor (Mesoscale) 2 km resolution & DQFs",
    "ABI-L2-CMIPC": \
            "ABI Level 2 Cloud and Moisture Imagery CONUS",
    "ABI-L2-CMIPF": \
            "ABI Level 2 Cloud and Moisture Imagery Full Disk",
    "ABI-L2-CMIPM": \
            "ABI Level 2 Cloud and Moisture Imagery Mesoscale",
    "ABI-L2-CODC": \
            "ABI Level 2 Cloud Optical Depth CONUS",
    "ABI-L2-CODF": \
            "ABI Level 2 Cloud Optical Depth Full Disk",
    "ABI-L2-CPSC": \
            "ABI Level 2 Cloud Particle Size CONUS",
    "ABI-L2-CPSF": \
            "ABI Level 2 Cloud Particle Size Full Disk",
    "ABI-L2-CPSM": \
            "ABI Level 2 Cloud Particle Size Mesoscale",
    "ABI-L2-CTPC": \
            "ABI Level 2 Cloud Top Pressure CONUS",
    "ABI-L2-CTPF": \
            "ABI Level 2 Cloud Top Pressure Full Disk",
    "ABI-L2-DMWC": \
            "ABI Level 2 Derived Motion Winds CONUS",
    "ABI-L2-DMWF": \
            "ABI Level 2 Derived Motion Winds Full Disk",
    "ABI-L2-DMWM": \
            "ABI Level 2 Derived Motion Winds Mesoscale",
    "ABI-L2-DMWVC": \
            "L2+ Derived Motion Winds - Vapor CONUS",
    "ABI-L2-DMWVF": \
            "L2+ Derived Motion Winds - Vapor Full Disk",
    "ABI-L2-DMWVF": \
            "L2+ Derived Motion Winds - Vapor Mesoscale",
    "ABI-L2-DSIC": \
            "ABI Level 2 Derived Stability Indices CONUS",
    "ABI-L2-DSIF": \
            "ABI Level 2 Derived Stability Indices Full Disk",
    "ABI-L2-DSIM": \
            "ABI Level 2 Derived Stability Indices Mesoscale",
    "ABI-L2-DSRC": \
            "ABI Level 2 Downward Shortwave Radiation CONUS",
    "ABI-L2-DSRF": \
            "ABI Level 2 Downward Shortwave Radiation Full Disk",
    "ABI-L2-DSRM": \
            "ABI Level 2 Downward Shortwave Radiation Mesoscale",
    "ABI-L2-FDCC": \
            "ABI Level 2 Fire (Hot Spot Characterization) CONUS",
    "ABI-L2-FDCF": \
            "ABI Level 2 Fire (Hot Spot Characterization) Full Disk",
    "ABI-L2-FDCM": \
            "ABI Level 2 Fire (Hot Spot Characterization) Mesoscale",
    "ABI-L2-LSAC": \
            "Land Surface Albedo (CONUS) 2km resolution & DQFs",
    "ABI-L2-LSAF": \
            "Land Surface Albedo (Full Disk) 2km resolution & DQFs",
    "ABI-L2-LSAM": \
            "Land Surface Albedo (Mesoscale) 2km resolution & DQFs",
    "ABI-L2-LSTC": \
            "ABI Level 2 Land Surface Temperature CONUS",
    "ABI-L2-LSTF": \
            "ABI Level 2 Land Surface Temperature Full Disk",
    "ABI-L2-LSTM": \
            "ABI Level 2 Land Surface Temperature Mesoscale",
    "ABI-L2-LVMPC": \
            "ABI Level 2 Legacy Vertical Moisture Profile CONUS",
    "ABI-L2-LVMPF": \
            "ABI Level 2 Legacy Vertical Moisture Profile Full Disk",
    "ABI-L2-LVMPM": \
            "ABI Level 2 Legacy Vertical Moisture Profile Mesoscale",
    "ABI-L2-LVTPC": \
            "ABI Level 2 Legacy Vertical Temperature Profile CONUS",
    "ABI-L2-LVTPF": \
            "ABI Level 2 Legacy Vertical Temperature Profile Full Disk",
    "ABI-L2-LVTPM": \
            "ABI Level 2 Legacy Vertical Temperature Profile Mesoscale",
    "ABI-L2-MCMIPC": \
            "ABI Level 2 Cloud and Moisture Imagery CONUS",
    "ABI-L2-MCMIPF": \
            "ABI Level 2 Cloud and Moisture Imagery Full Disk",
    "ABI-L2-MCMIPM": \
            "ABI Level 2 Cloud and Moisture Imagery Mesoscale",
    "ABI-L2-RRQPEF": \
            "ABI Level 2 Rainfall Rate (Quantitative Precipitation Estimate) Full Disk",
    "ABI-L2-RSRC": \
            "ABI Level 2 Reflected Shortwave Radiation Top-Of-Atmosphere CONUS",
    "ABI-L2-RSRF": \
            "ABI Level 2 Reflected Shortwave Radiation Top-Of-Atmosphere Full Disk",
    "ABI-L2-SSTF": \
            "ABI Level 2 Sea Surface (Skin) Temperature Full Disk",
    "ABI-L2-TPWC": \
            "ABI Level 2 Total Precipitable Water CONUS",
    "ABI-L2-TPWF": \
            "ABI Level 2 Total Precipitable Water Full Disk",
    "ABI-L2-TPWM": \
            "ABI Level 2 Total Precipitable Water Mesoscale",
    "ABI-L2-VAAF": \
            "ABI Level 2 Volcanic Ash: Detection and Height Full Disk",
    "EXIS-L1b-SFEU": \
            "EXIS-Solar Flux: EUV",
    "EXIS-L1b-SFXR": \
            "EXIS-Solar Flux: X-Ray",
    "GLM-L2-LCFA": \
            "GLM Level 2 Lightning Detection",
    "MAG-L1b-GEOF": \
            "MAG-Geomagnetic Field",
    "SEIS-L1b-EHIS": \
            "SEISS-Energetic Heavy Ions",
    "SEIS-L1b-MPSH": \
            "SEISS-Mag. Electrons & Protons: Med & High Energy",
    "SEIS-L1b-MPSL": \
            "SEISS-Mag. Electrons & Protons: Low Energy",
    "SEIS-L1b-SGPS": \
            "SEISS-Solar & Galactic Protons",
    "SUVI-L1b-Fe093": \
            "SUVI Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe131": \
            "SUVI Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe171": \
            "SUVI Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe195": \
            "SUVI Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe284": \
            "SUVI Level 1b Extreme Ultraviolet",
    "SUVI-L1b-He303": \
            "SUVI Level 1b Extreme Ultraviolet",
    }

goes_products = [
        ('17', 'ABI', 'L1b', 'RadC'),
        ('17', 'ABI', 'L1b', 'RadF'),
        ('17', 'ABI', 'L1b', 'RadM'),
        ('17', 'ABI', 'L2', 'ACHAC'),
        ('17', 'ABI', 'L2', 'ACHAF'),
        ('17', 'ABI', 'L2', 'ACHAM'),
        ('17', 'ABI', 'L2', 'ACHTF'),
        ('17', 'ABI', 'L2', 'ACHTM'),
        ('17', 'ABI', 'L2', 'ACMC'),
        ('17', 'ABI', 'L2', 'ACMF'),
        ('17', 'ABI', 'L2', 'ACMM'),
        ('17', 'ABI', 'L2', 'ACTPC'),
        ('17', 'ABI', 'L2', 'ACTPF'),
        ('17', 'ABI', 'L2', 'ACTPM'),
        ('17', 'ABI', 'L2', 'ADPC'),
        ('17', 'ABI', 'L2', 'ADPF'),
        ('17', 'ABI', 'L2', 'ADPM'),
        ('17', 'ABI', 'L2', 'AICEF'),
        ('17', 'ABI', 'L2', 'AITAF'),
        ('17', 'ABI', 'L2', 'AODC'),
        ('17', 'ABI', 'L2', 'AODF'),
        ('17', 'ABI', 'L2', 'BRFC'),
        ('17', 'ABI', 'L2', 'BRFF'),
        ('17', 'ABI', 'L2', 'BRFM'),
        ('17', 'ABI', 'L2', 'CMIPC'),
        ('17', 'ABI', 'L2', 'CMIPF'),
        ('17', 'ABI', 'L2', 'CMIPM'),
        ('17', 'ABI', 'L2', 'CODC'),
        ('17', 'ABI', 'L2', 'CODF'),
        ('17', 'ABI', 'L2', 'CPSC'),
        ('17', 'ABI', 'L2', 'CPSF'),
        ('17', 'ABI', 'L2', 'CPSM'),
        ('17', 'ABI', 'L2', 'CTPC'),
        ('17', 'ABI', 'L2', 'CTPF'),
        ('17', 'ABI', 'L2', 'DMWC'),
        ('17', 'ABI', 'L2', 'DMWF'),
        ('17', 'ABI', 'L2', 'DMWM'),
        ('17', 'ABI', 'L2', 'DMWVC'),
        ('17', 'ABI', 'L2', 'DMWVF'),
        ('17', 'ABI', 'L2', 'DMWVM'),
        ('17', 'ABI', 'L2', 'DSIC'),
        ('17', 'ABI', 'L2', 'DSIF'),
        ('17', 'ABI', 'L2', 'DSIM'),
        ('17', 'ABI', 'L2', 'DSRC'),
        ('17', 'ABI', 'L2', 'DSRF'),
        ('17', 'ABI', 'L2', 'DSRM'),
        ('17', 'ABI', 'L2', 'FDCC'),
        ('17', 'ABI', 'L2', 'FDCF'),
        ('17', 'ABI', 'L2', 'FDCM'),
        ('17', 'ABI', 'L2', 'LSAC'),
        ('17', 'ABI', 'L2', 'LSAF'),
        ('17', 'ABI', 'L2', 'LSAM'),
        ('17', 'ABI', 'L2', 'LST2KMF'),
        ('17', 'ABI', 'L2', 'LSTC'),
        ('17', 'ABI', 'L2', 'LSTF'),
        ('17', 'ABI', 'L2', 'LSTM'),
        ('17', 'ABI', 'L2', 'LVMPC'),
        ('17', 'ABI', 'L2', 'LVMPF'),
        ('17', 'ABI', 'L2', 'LVMPM'),
        ('17', 'ABI', 'L2', 'LVTPC'),
        ('17', 'ABI', 'L2', 'LVTPF'),
        ('17', 'ABI', 'L2', 'LVTPM'),
        ('17', 'ABI', 'L2', 'MCMIPC'),
        ('17', 'ABI', 'L2', 'MCMIPF'),
        ('17', 'ABI', 'L2', 'MCMIPM'),
        ('17', 'ABI', 'L2', 'RRQPEF'),
        ('17', 'ABI', 'L2', 'RSRC'),
        ('17', 'ABI', 'L2', 'RSRF'),
        ('17', 'ABI', 'L2', 'SSTF'),
        ('17', 'ABI', 'L2', 'TPWC'),
        ('17', 'ABI', 'L2', 'TPWF'),
        ('17', 'ABI', 'L2', 'TPWM'),
        ('17', 'ABI', 'L2', 'VAAF'),
        ('17', 'EXIS', 'L1b', 'SFEU'),
        ('17', 'EXIS', 'L1b', 'SFXR'),
        ('17', 'GLM', 'L2', 'LCFA'),
        ('17', 'MAG', 'L1b', 'GEOF'),
        ('17', 'SEIS', 'L1b', 'EHIS'),
        ('17', 'SEIS', 'L1b', 'MPSH'),
        ('17', 'SEIS', 'L1b', 'MPSL'),
        ('17', 'SEIS', 'L1b', 'SGPS'),
        ('17', 'SUVI', 'L1b', 'Fe093'),
        ('17', 'SUVI', 'L1b', 'Fe131'),
        ('17', 'SUVI', 'L1b', 'Fe171'),
        ('17', 'SUVI', 'L1b', 'Fe195'),
        ('17', 'SUVI', 'L1b', 'Fe284'),
        ('17', 'SUVI', 'L1b', 'He303'),
        ('18', 'ABI', 'L1b', 'RadC'),
        ('18', 'ABI', 'L1b', 'RadF'),
        ('18', 'ABI', 'L1b', 'RadM'),
        ('18', 'ABI', 'L2', 'ACHA2KMC'),
        ('18', 'ABI', 'L2', 'ACHA2KMF'),
        ('18', 'ABI', 'L2', 'ACHA2KMM'),
        ('18', 'ABI', 'L2', 'ACHAC'),
        ('18', 'ABI', 'L2', 'ACHAF'),
        ('18', 'ABI', 'L2', 'ACHAM'),
        ('18', 'ABI', 'L2', 'ACHP2KMC'),
        ('18', 'ABI', 'L2', 'ACHP2KMF'),
        ('18', 'ABI', 'L2', 'ACHP2KMM'),
        ('18', 'ABI', 'L2', 'ACHTF'),
        ('18', 'ABI', 'L2', 'ACHTM'),
        ('18', 'ABI', 'L2', 'ACMC'),
        ('18', 'ABI', 'L2', 'ACMF'),
        ('18', 'ABI', 'L2', 'ACMM'),
        ('18', 'ABI', 'L2', 'ACTPC'),
        ('18', 'ABI', 'L2', 'ACTPF'),
        ('18', 'ABI', 'L2', 'ACTPM'),
        ('18', 'ABI', 'L2', 'ADPC'),
        ('18', 'ABI', 'L2', 'ADPF'),
        ('18', 'ABI', 'L2', 'ADPM'),
        ('18', 'ABI', 'L2', 'AICEF'),
        ('18', 'ABI', 'L2', 'AITAF'),
        ('18', 'ABI', 'L2', 'AODC'),
        ('18', 'ABI', 'L2', 'AODF'),
        ('18', 'ABI', 'L2', 'BRFC'),
        ('18', 'ABI', 'L2', 'BRFF'),
        ('18', 'ABI', 'L2', 'BRFM'),
        ('18', 'ABI', 'L2', 'CCLC'),
        ('18', 'ABI', 'L2', 'CCLF'),
        ('18', 'ABI', 'L2', 'CCLM'),
        ('18', 'ABI', 'L2', 'CMIPC'),
        ('18', 'ABI', 'L2', 'CMIPF'),
        ('18', 'ABI', 'L2', 'CMIPM'),
        ('18', 'ABI', 'L2', 'COD2KMF'),
        ('18', 'ABI', 'L2', 'CODC'),
        ('18', 'ABI', 'L2', 'CODF'),
        ('18', 'ABI', 'L2', 'CPSC'),
        ('18', 'ABI', 'L2', 'CPSF'),
        ('18', 'ABI', 'L2', 'CPSM'),
        ('18', 'ABI', 'L2', 'CTPC'),
        ('18', 'ABI', 'L2', 'CTPF'),
        ('18', 'ABI', 'L2', 'DMWC'),
        ('18', 'ABI', 'L2', 'DMWF'),
        ('18', 'ABI', 'L2', 'DMWM'),
        ('18', 'ABI', 'L2', 'DMWVC'),
        ('18', 'ABI', 'L2', 'DMWVF'),
        ('18', 'ABI', 'L2', 'DMWVM'),
        ('18', 'ABI', 'L2', 'DSIC'),
        ('18', 'ABI', 'L2', 'DSIF'),
        ('18', 'ABI', 'L2', 'DSIM'),
        ('18', 'ABI', 'L2', 'DSRC'),
        ('18', 'ABI', 'L2', 'DSRF'),
        ('18', 'ABI', 'L2', 'DSRM'),
        ('18', 'ABI', 'L2', 'FDCC'),
        ('18', 'ABI', 'L2', 'FDCF'),
        ('18', 'ABI', 'L2', 'FDCM'),
        ('18', 'ABI', 'L2', 'FSCC'),
        ('18', 'ABI', 'L2', 'FSCF'),
        ('18', 'ABI', 'L2', 'FSCM'),
        ('18', 'ABI', 'L2', 'LSAC'),
        ('18', 'ABI', 'L2', 'LSAF'),
        ('18', 'ABI', 'L2', 'LSAM'),
        ('18', 'ABI', 'L2', 'LST2KMF'),
        ('18', 'ABI', 'L2', 'LSTC'),
        ('18', 'ABI', 'L2', 'LSTF'),
        ('18', 'ABI', 'L2', 'LSTM'),
        ('18', 'ABI', 'L2', 'LVMPC'),
        ('18', 'ABI', 'L2', 'LVMPF'),
        ('18', 'ABI', 'L2', 'LVMPM'),
        ('18', 'ABI', 'L2', 'LVTPC'),
        ('18', 'ABI', 'L2', 'LVTPF'),
        ('18', 'ABI', 'L2', 'LVTPM'),
        ('18', 'ABI', 'L2', 'MCMIPC'),
        ('18', 'ABI', 'L2', 'MCMIPF'),
        ('18', 'ABI', 'L2', 'MCMIPM'),
        ('18', 'ABI', 'L2', 'RRQPEF'),
        ('18', 'ABI', 'L2', 'RSRC'),
        ('18', 'ABI', 'L2', 'RSRF'),
        ('18', 'ABI', 'L2', 'SSTF'),
        ('18', 'ABI', 'L2', 'TPWC'),
        ('18', 'ABI', 'L2', 'TPWF'),
        ('18', 'ABI', 'L2', 'TPWM'),
        ('18', 'EXIS', 'L1b', 'SFEU'),
        ('18', 'EXIS', 'L1b', 'SFXR'),
        ('18', 'GLM', 'L2', 'LCFA'),
        ('18', 'MAG', 'L1b', 'GEOF'),
        ('18', 'SEIS', 'L1b', 'EHIS'),
        ('18', 'SEIS', 'L1b', 'MPSH'),
        ('18', 'SEIS', 'L1b', 'MPSL'),
        ('18', 'SEIS', 'L1b', 'SGPS'),
        ('18', 'SUVI', 'L1b', 'Fe093'),
        ('18', 'SUVI', 'L1b', 'Fe131'),
        ('18', 'SUVI', 'L1b', 'Fe171'),
        ('18', 'SUVI', 'L1b', 'Fe195'),
        ('18', 'SUVI', 'L1b', 'Fe284'),
        ('18', 'SUVI', 'L1b', 'He303'),
        ('16', 'ABI', 'L1b', 'RadC'),
        ('16', 'ABI', 'L1b', 'RadF'),
        ('16', 'ABI', 'L1b', 'RadM'),
        ('16', 'ABI', 'L2', 'ACHA2KMC'),
        ('16', 'ABI', 'L2', 'ACHA2KMF'),
        ('16', 'ABI', 'L2', 'ACHA2KMM'),
        ('16', 'ABI', 'L2', 'ACHAC'),
        ('16', 'ABI', 'L2', 'ACHAF'),
        ('16', 'ABI', 'L2', 'ACHAM'),
        ('16', 'ABI', 'L2', 'ACHP2KMC'),
        ('16', 'ABI', 'L2', 'ACHP2KMF'),
        ('16', 'ABI', 'L2', 'ACHP2KMM'),
        ('16', 'ABI', 'L2', 'ACHTF'),
        ('16', 'ABI', 'L2', 'ACHTM'),
        ('16', 'ABI', 'L2', 'ACMC'),
        ('16', 'ABI', 'L2', 'ACMF'),
        ('16', 'ABI', 'L2', 'ACMM'),
        ('16', 'ABI', 'L2', 'ACTPC'),
        ('16', 'ABI', 'L2', 'ACTPF'),
        ('16', 'ABI', 'L2', 'ACTPM'),
        ('16', 'ABI', 'L2', 'ADPC'),
        ('16', 'ABI', 'L2', 'ADPF'),
        ('16', 'ABI', 'L2', 'ADPM'),
        ('16', 'ABI', 'L2', 'AICEF'),
        ('16', 'ABI', 'L2', 'AITAF'),
        ('16', 'ABI', 'L2', 'AODC'),
        ('16', 'ABI', 'L2', 'AODF'),
        ('16', 'ABI', 'L2', 'BRFC'),
        ('16', 'ABI', 'L2', 'BRFF'),
        ('16', 'ABI', 'L2', 'BRFM'),
        ('16', 'ABI', 'L2', 'CCLC'),
        ('16', 'ABI', 'L2', 'CCLF'),
        ('16', 'ABI', 'L2', 'CCLM'),
        ('16', 'ABI', 'L2', 'CMIPC'),
        ('16', 'ABI', 'L2', 'CMIPF'),
        ('16', 'ABI', 'L2', 'CMIPM'),
        ('16', 'ABI', 'L2', 'COD2KMF'),
        ('16', 'ABI', 'L2', 'CODC'),
        ('16', 'ABI', 'L2', 'CODF'),
        ('16', 'ABI', 'L2', 'CPSC'),
        ('16', 'ABI', 'L2', 'CPSF'),
        ('16', 'ABI', 'L2', 'CPSM'),
        ('16', 'ABI', 'L2', 'CTPC'),
        ('16', 'ABI', 'L2', 'CTPF'),
        ('16', 'ABI', 'L2', 'DMWC'),
        ('16', 'ABI', 'L2', 'DMWF'),
        ('16', 'ABI', 'L2', 'DMWM'),
        ('16', 'ABI', 'L2', 'DMWVC'),
        ('16', 'ABI', 'L2', 'DMWVF'),
        ('16', 'ABI', 'L2', 'DMWVM'),
        ('16', 'ABI', 'L2', 'DSIC'),
        ('16', 'ABI', 'L2', 'DSIF'),
        ('16', 'ABI', 'L2', 'DSIM'),
        ('16', 'ABI', 'L2', 'DSRC'),
        ('16', 'ABI', 'L2', 'DSRF'),
        ('16', 'ABI', 'L2', 'DSRM'),
        ('16', 'ABI', 'L2', 'FDCC'),
        ('16', 'ABI', 'L2', 'FDCF'),
        ('16', 'ABI', 'L2', 'FDCM'),
        ('16', 'ABI', 'L2', 'FSCC'),
        ('16', 'ABI', 'L2', 'FSCF'),
        ('16', 'ABI', 'L2', 'FSCM'),
        ('16', 'ABI', 'L2', 'LSAC'),
        ('16', 'ABI', 'L2', 'LSAF'),
        ('16', 'ABI', 'L2', 'LSAM'),
        ('16', 'ABI', 'L2', 'LST2KMF'),
        ('16', 'ABI', 'L2', 'LSTC'),
        ('16', 'ABI', 'L2', 'LSTF'),
        ('16', 'ABI', 'L2', 'LSTM'),
        ('16', 'ABI', 'L2', 'LVMPC'),
        ('16', 'ABI', 'L2', 'LVMPF'),
        ('16', 'ABI', 'L2', 'LVMPM'),
        ('16', 'ABI', 'L2', 'LVTPC'),
        ('16', 'ABI', 'L2', 'LVTPF'),
        ('16', 'ABI', 'L2', 'LVTPM'),
        ('16', 'ABI', 'L2', 'MCMIPC'),
        ('16', 'ABI', 'L2', 'MCMIPF'),
        ('16', 'ABI', 'L2', 'MCMIPM'),
        ('16', 'ABI', 'L2', 'RRQPEF'),
        ('16', 'ABI', 'L2', 'RSRC'),
        ('16', 'ABI', 'L2', 'RSRF'),
        ('16', 'ABI', 'L2', 'SSTF'),
        ('16', 'ABI', 'L2', 'TPWC'),
        ('16', 'ABI', 'L2', 'TPWF'),
        ('16', 'ABI', 'L2', 'TPWM'),
        ('16', 'ABI', 'L2', 'VAAF'),
        ('16', 'EXIS', 'L1b', 'SFEU'),
        ('16', 'EXIS', 'L1b', 'SFXR'),
        ('16', 'GLM', 'L2', 'LCFA'),
        ('16', 'MAG', 'L1b', 'GEOF'),
        ('16', 'SEIS', 'L1b', 'EHIS'),
        ('16', 'SEIS', 'L1b', 'MPSH'),
        ('16', 'SEIS', 'L1b', 'MPSL'),
        ('16', 'SEIS', 'L1b', 'SGPS'),
        ('16', 'SUVI', 'L1b', 'Fe093'),
        ('16', 'SUVI', 'L1b', 'Fe131'),
        ('16', 'SUVI', 'L1b', 'Fe171'),
        ('16', 'SUVI', 'L1b', 'Fe195'),
        ('16', 'SUVI', 'L1b', 'Fe284'),
        ('16', 'SUVI', 'L1b', 'He303'),
        ]
