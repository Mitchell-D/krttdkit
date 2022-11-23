from pprint import pprint as ppt
from dataclasses import dataclass

"""
Dictionary of AWS bucket ABI product keys and descriptions,
obtained 11/22/2022 from:
https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16
"""
abi = {
    "ABI-L1b-RadF": "Advanced Baseline Imager Level 1b Full Disk",
    "ABI-L1b-RadC": "Advanced Baseline Imager Level 1b CONUS",
    "ABI-L1b-RadM": "Advanced Baseline Imager Level 1b Mesoscale",
    "ABI-L2-ACHAC": "Advanced Baseline Imager Level 2 Cloud Top Height CONUS",
    "ABI-L2-ACHAF": "Advanced Baseline Imager Level 2 Cloud Top Height Full Disk",
    "ABI-L2-ACHAM": "Advanced Baseline Imager Level 2 Cloud Top Height Mesoscale",
    "ABI-L2-ACHTF": "Advanced Baseline Imager Level 2 Cloud Top Temperature Full Disk",
    "ABI-L2-ACHTM": "Advanced Baseline Imager Level 2 Cloud Top Temperature Mesoscale",
    "ABI-L2-ACMC": "Advanced Baseline Imager Level 2 Clear Sky Mask CONUS",
    "ABI-L2-ACMF": "Advanced Baseline Imager Level 2 Clear Sky Mask Full Disk",
    "ABI-L2-ACMM": "Advanced Baseline Imager Level 2 Clear Sky Mask Mesoscale",
    "ABI-L2-ACTPC": "Advanced Baseline Imager Level 2 Cloud Top Phase CONUS",
    "ABI-L2-ACTPF": "Advanced Baseline Imager Level 2 Cloud Top Phase Full Disk",
    "ABI-L2-ACTPM": "Advanced Baseline Imager Level 2 Cloud Top Phase Mesoscale",
    "ABI-L2-ADPC": "Advanced Baseline Imager Level 2 Aerosol Detection CONUS",
    "ABI-L2-ADPF": "Advanced Baseline Imager Level 2 Aerosol Detection Full Disk",
    "ABI-L2-ADPM": "Advanced Baseline Imager Level 2 Aerosol Detection Mesoscale",
    "ABI-L2-AICEF": "Ice Concentration and Extent",
    "ABI-L2-AITAF": "Ice Age and Thickness",
    "ABI-L2-AODC": "Advanced Baseline Imager Level 2 Aerosol Optical Depth CONUS",
    "ABI-L2-AODF": "Advanced Baseline Imager Level 2 Aerosol Optical Depth Full Disk",
    "ABI-L2-BRFC": "Land Surface Bidirectional Reflectance Factor (CONUS) 2 km resolution & DQFs",
    "ABI-L2-BRFF": "Land Surface Bidirectional Reflectance Factor (Full Disk) 2 km resolution & DQFs",
    "ABI-L2-BRFM": "Land Surface Bidirectional Reflectance Factor (Mesoscale) 2 km resolution & DQFs",
    "ABI-L2-CMIPC": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery CONUS",
    "ABI-L2-CMIPF": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery Full Disk",
    "ABI-L2-CMIPM": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery Mesoscale",
    "ABI-L2-CODC": "Advanced Baseline Imager Level 2 Cloud Optical Depth CONUS",
    "ABI-L2-CODF": "Advanced Baseline Imager Level 2 Cloud Optical Depth Full Disk",
    "ABI-L2-CPSC": "Advanced Baseline Imager Level 2 Cloud Particle Size CONUS",
    "ABI-L2-CPSF": "Advanced Baseline Imager Level 2 Cloud Particle Size Full Disk",
    "ABI-L2-CPSM": "Advanced Baseline Imager Level 2 Cloud Particle Size Mesoscale",
    "ABI-L2-CTPC": "Advanced Baseline Imager Level 2 Cloud Top Pressure CONUS",
    "ABI-L2-CTPF": "Advanced Baseline Imager Level 2 Cloud Top Pressure Full Disk",
    "ABI-L2-DMWC": "Advanced Baseline Imager Level 2 Derived Motion Winds CONUS",
    "ABI-L2-DMWF": "Advanced Baseline Imager Level 2 Derived Motion Winds Full Disk",
    "ABI-L2-DMWM": "Advanced Baseline Imager Level 2 Derived Motion Winds Mesoscale",
    "ABI-L2-DMWVC": "L2+ Derived Motion Winds - Vapor CONUS",
    "ABI-L2-DMWVF": "L2+ Derived Motion Winds - Vapor Full Disk",
    "ABI-L2-DMWVF": "L2+ Derived Motion Winds - Vapor Mesoscale",
    "ABI-L2-DSIC": "Advanced Baseline Imager Level 2 Derived Stability Indices CONUS",
    "ABI-L2-DSIF": "Advanced Baseline Imager Level 2 Derived Stability Indices Full Disk",
    "ABI-L2-DSIM": "Advanced Baseline Imager Level 2 Derived Stability Indices Mesoscale",
    "ABI-L2-DSRC": "Advanced Baseline Imager Level 2 Downward Shortwave Radiation CONUS",
    "ABI-L2-DSRF": "Advanced Baseline Imager Level 2 Downward Shortwave Radiation Full Disk",
    "ABI-L2-DSRM": "Advanced Baseline Imager Level 2 Downward Shortwave Radiation Mesoscale",
    "ABI-L2-FDCC": "Advanced Baseline Imager Level 2 Fire (Hot Spot Characterization) CONUS",
    "ABI-L2-FDCF": "Advanced Baseline Imager Level 2 Fire (Hot Spot Characterization) Full Disk",
    "ABI-L2-FDCM": "Advanced Baseline Imager Level 2 Fire (Hot Spot Characterization) Mesoscale",
    "ABI-L2-LSAC": "Land Surface Albedo (CONUS) 2km resolution & DQFs",
    "ABI-L2-LSAF": "Land Surface Albedo (Full Disk) 2km resolution & DQFs",
    "ABI-L2-LSAM": "Land Surface Albedo (Mesoscale) 2km resolution & DQFs",
    "ABI-L2-LSTC": "Advanced Baseline Imager Level 2 Land Surface Temperature CONUS",
    "ABI-L2-LSTF": "Advanced Baseline Imager Level 2 Land Surface Temperature Full Disk",
    "ABI-L2-LSTM": "Advanced Baseline Imager Level 2 Land Surface Temperature Mesoscale",
    "ABI-L2-LVMPC": "Advanced Baseline Imager Level 2 Legacy Vertical Moisture Profile CONUS",
    "ABI-L2-LVMPF": "Advanced Baseline Imager Level 2 Legacy Vertical Moisture Profile Full Disk",
    "ABI-L2-LVMPM": "Advanced Baseline Imager Level 2 Legacy Vertical Moisture Profile Mesoscale",
    "ABI-L2-LVTPC": "Advanced Baseline Imager Level 2 Legacy Vertical Temperature Profile CONUS",
    "ABI-L2-LVTPF": "Advanced Baseline Imager Level 2 Legacy Vertical Temperature Profile Full Disk",
    "ABI-L2-LVTPM": "Advanced Baseline Imager Level 2 Legacy Vertical Temperature Profile Mesoscale",
    "ABI-L2-MCMIPC": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery CONUS",
    "ABI-L2-MCMIPF": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery Full Disk",
    "ABI-L2-MCMIPM": "Advanced Baseline Imager Level 2 Cloud and Moisture Imagery Mesoscale",
    "ABI-L2-RRQPEF": "Advanced Baseline Imager Level 2 Rainfall Rate (Quantitative Precipitation Estimate) Full Disk",
    "ABI-L2-RSRC": "Advanced Baseline Imager Level 2 Reflected Shortwave Radiation Top-Of-Atmosphere CONUS",
    "ABI-L2-RSRF": "Advanced Baseline Imager Level 2 Reflected Shortwave Radiation Top-Of-Atmosphere Full Disk",
    "ABI-L2-SSTF": "Advanced Baseline Imager Level 2 Sea Surface (Skin) Temperature Full Disk",
    "ABI-L2-TPWC": "Advanced Baseline Imager Level 2 Total Precipitable Water CONUS",
    "ABI-L2-TPWF": "Advanced Baseline Imager Level 2 Total Precipitable Water Full Disk",
    "ABI-L2-TPWM": "Advanced Baseline Imager Level 2 Total Precipitable Water Mesoscale",
    "ABI-L2-VAAF": "Advanced Baseline Imager Level 2 Volcanic Ash: Detection and Height Full Disk",
    "EXIS-L1b-SFEU": "EXIS-Solar Flux: EUV",
    "EXIS-L1b-SFXR": "EXIS-Solar Flux: X-Ray",
    "GLM-L2-LCFA": "Geostationary Lightning Mapper Level 2 Lightning Detection",
    "MAG-L1b-GEOF": "MAG-Geomagnetic Field",
    "SEIS-L1b-EHIS": "SEISS-Energetic Heavy Ions",
    "SEIS-L1b-MPSH": "SEISS-Mag. Electrons & Protons: Med & High Energy",
    "SEIS-L1b-MPSL": "SEISS-Mag. Electrons & Protons: Low Energy",
    "SEIS-L1b-SGPS": "SEISS-Solar & Galactic Protons",
    "SUVI-L1b-Fe093": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe131": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe171": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe195": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    "SUVI-L1b-Fe284": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    "SUVI-L1b-He303": "Solar Ultraviolet Imager Level 1b Extreme Ultraviolet",
    }

class ABIProduct:
    """
    ABIProduct is a hashable object that describes a specific ABI product
    based on instrument, level, and product keys.
    """
    domains = { "Solar", "Full Disk", "CONUS", "Mesoscale" }
    def __init__(self, instrument:str, level:str, product:str,
                 description:str=""):
        self._instrument = instrument
        self._level = level
        self._product = product
        self._description = description
        self._domain = None

    def __hash__(self):
        """ ABIProducts are hashed by their aws string """
        return hash(str(self.aws_string))
    @staticmethod
    def from_aws_string(key):
        return ABIProduct(*tuple(key.split("-")))
    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, description):
        self._description = description
        self._domain = next(( d for d in ABIProduct.domains
                         if d in self.description ), None)
    @property
    def instrument(self):
        return self._instrument
    @property
    def level(self):
        return self._level
    @property
    def product(self):
        return self._product
    @property
    def aws_string(self):
        return "-".join((self._instrument, self._level, self._product))
    @property
    def domain(self):
        """ Domain name inferred from keywords in description """
        return self._domain
    def __repr__(self):
        return self.aws_string

def get_products():
    """
    Returns a list of products and descriptions based on the dictionary
    """
    keys = list(abi.keys())
    products = [ ABIProduct.from_aws_string(key) for key in keys ]
    for p in products:
        p.description = abi[p.aws_string]
    return products

def findall(mapping:dict={}, products:list=None):
    """
    Helper method for finding information about ABI products.

    Returns all ABIProduct objects in the provided list with attribute that
    match the key:value pairs in the mapping parameter.

    This relies on the ABIProduct convention of prefacing the hidden attributes
    associated with each property with an underscore.

    :param products: list of ABIProduct objects to check from. If None, a new
            list of all products is generated
    :param mapping: dictionary of mappings from attribute names to a str value.
            For example, mapping={"level":"L2"} will return all products in
            the provided list with "level" property set to "L2"
    """
    # Get the ABIProduct objects if they weren't provided
    if products is None:
        products = get_products()
    mapping = dict(filter(lambda i: not i[1] is None, mapping.items()))
    matches = { k:[] for k in mapping.keys() }
    for p in products:
        for k in mapping.keys():
            if getattr(p, "_"+k) == mapping[k]:
                matches[k].append(p)
    return list(set.intersection(*[set(matches[k]) for k in matches.keys()]))

def menu():
    """
    Returns sets of options for domain, instrument, level, and product,
    and lists the keys/descriptsions of products without an associated domain.
    """
    products = get_products()
    return {
        "domain":{p.domain for p in products},
        "instrument":{p.instrument for p in products},
        "level":{p.level for p in products},
        "product":{p.product for p in products},
        "no_domain":[(p.aws_string, p.description) for p in products
                     if p.domain is None] }

if __name__=="__main__":
    ppt(menu())
    exit(0)
    descriptions = {}
    for p in products:
        key = p.aws_string
        if p.instrument not in descriptions.keys():
            descriptions[p.instrument] = {}
        if p.level not in descriptions[p.instrument].keys():
            descriptions[p.instrument][p.level] = {}
        descriptions[p.instrument][p.level][p.product] = abi[key]
