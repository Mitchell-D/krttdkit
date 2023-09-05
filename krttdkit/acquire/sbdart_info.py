default_params = [
    ('idatm',  4, 'Atmospheric profile ID'),
    ('amix',  0.0, 'Mixing factor between custom (atms.dat) and selected idatm profile'),
    ('isat',  0, 'Spectral response (filter) function ID'),
    ('wlinf',  0.550, 'Lower wavelength limit in um'),
    ('wlsup',  0.550, 'Upper wavelength limit in um'),
    ('wlinc',  0.0, 'Spectral resolution'),
    ('sza',  0.0, 'Solar zenith angle in deg'),
    ('csza',  -1.0, 'Cosine of solar zenith angle in deg'),
    ('solfac',  1.0, 'Solar distance factor'),
    ('nf',  2, 'Solar spectrum ID'),
    ('iday',  0, 'Day of year (for SZA calculation)'),
    ('time',  16.0, 'UTC time in decimal hours'),
    ('alat', -64.7670, 'Latitude of point on Earth\'s surface'),
    ('alon', -64.0670, 'Longitude of point on Earth\'s surface'),
    ('zpres',  -1.0, 'Surface altitude in km; alternative to PBAR'),
    # negative -> use original pressure profile
    ('pbar',  -1.0, 'Surface pressure in mb'),
    # negative -> use vertical profile
    ('sclh2o',  -1.0, 'Water vapor scale height (km)'),
    ('uw',  -1.0, 'Integrated water vapor (g/cm^2)'),
    ('uo3',  -1.0, 'Integrated ozone concentration (atm-cm)'),
    ('o3trp',  -1.0, 'Integrated tropospheric ozone concentration (atm-cm)'),
    # By default, UW sets integrated ozone for whole atmosphere with ZTRP=0.0
    ('ztrp',  0.0, 'Tropopause altitude'),
    ('xrsc',  1.0, 'Rayleigh scattering sensitivity'),
    ('xn2',  -1.0, 'N_2 volume mixing ratio (PPM)'),
    ('xo2',  -1.0, 'O_2 volume mixing ratio (PPM)'),
    ('xco2',  -1.0, 'CO_2 volume mixing ratio (PPM)'),
    ('xch4',  -1.0, 'CH_4 volume mixing ratio (PPM)'),
    ('xn2o',  -1.0, 'N_2O volume mixing ratio (PPM)'),
    ('xco',  -1.0, 'CO volume mixing ratio (PPM)'),
    ('xno2',  -1.0, 'NO_2 volume mixing ratio (PPM)'),
    ('xso2',  -1.0, 'SO_2 volume mixing ratio (PPM)'),
    ('xnh3',  -1.0, 'NH_3 volume mixing ratio (PPM)'),
    ('xno',  -1.0, 'NO volume mixing ratio (PPM)'),
    ('xhno3',  -1.0, 'HNO_3 volume mixing ratio (PPM)'),
    ('xo4',  1.0, 'Oxygen collisional complex absorption sensitivity'),
    ('isalb',  0, 'Surface albedo feature'),
    ('albcon',  0.0, 'Spectrally-uniform surface albedo'),
    ('sc', '1.0,3*0.0', 'Surface albedo params for ISALB in {7,8,10}'),
    ('zcloud',  '5*0.0', 'Cloud layer altitude in km (up to 5 values)'),
    ('tcloud',  '5*0.0', 'Cloud optical depth at 0.55um'),
    ('lwp',  '5*0.0', 'Liquid water path (g/m^2)'),
    ('nre',  '5*8.0', 'Cloud effective radius (um)'),
    # If RHCLD<0, relative humidity inside clouds follows profile
    ('rhcld',  -1.0, 'Relative humidity within cloud layers'),
    ('krhclr',  0, 'Clear-layer water vapor adjustment (if TCLOUD>0 or RHCLD>=0)'),
    ('jaer',  '5*0', '5-element array of stratospheric aerosol types'),
    ('zaer',  '5*0.0', 'Altitudes of stratospheric aerosol layers (km)'),
    ('taerst',  '5*0.0', 'Optical depth at 0.55um of stratospheric aerosol layers'),
    ('iaer', 0, 'Boundary layer aerosol ID'),
    ('vis',  23.0, 'Horizontal visibility due to aerosols at 0.55um (km)'),
    ('rhaer',  -1.0, 'Relative humidity for BL aerosol model (IAER)'),
    ('wlbaer',  '47*0.0', 'Wavelength points when IAER is 5 (um)'),
    # Significant for IAER=1,2,3,4
    ('tbaer',  '47*0.0', 'Vertical optical depth of BL aerosols at 0.55um'),
    ('abaer',  -1.0, 'Angstrom exp for BL aerosol extinction (if IAER is 5)'),
    ('wbaer',  '47*0.950', 'Single-scatter albedo (if IAER is 5)'),
    ('gbaer',  '47*0.70', 'Asymmetry factor (if IAER is 5)'),
    ('pmaer',  '940*0.0', 'Legendere moments of BL phase function (if IAER is 5)'),
    # Valid for positive IAER values
    ('zbaer',  '50*-1.0', 'Altitude grid for custom aerosol profile (km)'),
    # Valid for ZBAER grid points with arbitrary relative units
    ('dbaer',  '50*-1.0', 'Aerosol density at ZBAER points'),
    ('nothrm', -1, 'Thermal emission ID (-1, 0, or 1)'),
    ('nosct',  0, 'BL Aerosol scattering method ID'),
    ('kdist',  3, 'Transmission scheme ID'),
    ('zgrid1',  0.0, 'Lower-atmosphere resolution (km)'),
    ('zgrid2',  30.0, 'Upper-atmosphere resolution (km)'),
    ('ngrid',  50, 'Number of vertical grid points'),
    ('zout',  0.0,100.0, 'Bottom and top altitudes for IOUT (km)'),
    ('iout',  10, 'Output format ID'),
    ('deltam', 't', 'If True, Uses Delta-m method (Wiscombe, 1977)'),
    ('lamber',  't', ''),
    ('ibcnd',  0, 'Boundary illumination; 1 if isotropic illumination from bottom'),
    ('saza',  180.0, 'Solar azimuth angle (deg)'),
    ('prnt',  '7*f', 'DISORT output option ID'),
    ('ipth',  1, ''),
    ('fisot',  0.0, 'Top isotropic illumination (W/m^2)'),
    ('temis',  0.0, 'Top-layer emissivity'),
    ('nstr',  4, 'Number of internal radiation streams used'),
    ('nzen',  0, 'Number of viewing zenith angles'),
    ('uzen',  20*-1.0, 'Specific viewing zenith angles'),
    ('vzen',  20*90, 'User NADIR angles (ie 180-UZEN)'),
    ('nphi',  0, 'Number of viewing azimuth angles'),
    ('phi',  20*-1.0, 'Specific viewer azimuth angles'),
    ('imomc',  3, 'Cloud model phase function ID'),
    ('imoma',  3, 'BL Aerosol phase function ID'),
    ('ttemp',  -1.0, 'Top-layer thermal emission'),
    ('btemp',  -1.0, 'Bottom-layer thermal emission'),
    ('spowder',  'f', 'Additional surface layer scattering'),
    ('idb',  20*0, 'Diagnostic output ID'),
    ]
