
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta

from krttdkit.acquire import get_goes as gg
from krttdkit.acquire import abi
from krttdkit.acquire import gesdisc
from krttdkit.products import FeatureGrid
import krttdkit.visualize.guitools as gt
import krttdkit.operate.enhance as enh

def get_nldas_urls(t0:datetime, tf:datetime):
    """
    """
    print(gesdisc.hourly_nldas2_urls(t0=t0, tf=tf))

if __name__=="__main__":
    # Cloud top height product
    #goesapi = gg.GetGOES()
    #gg.list_config()

    target_time = datetime(2023,8,7,17,47)
    time_window = timedelta(hours=1)
    data_dir = Path("data/abi_l2")

    """ Get a L2 file for:
     (1) achac: cloud top height
     (2) acmc:  clear sky mask
    """
    l2_2km_prods = ["ACMC", "ACTPC"]
    l2_2km_prods = [
            abi.download_l2_abi(data_dir,p,target_time, satellite="18")[0]
            for p in l2_2km_prods]

    # Only take the first (earliest) file

    """ Get a NLDAS-2 file for the target time """
    #get_nldas_urls(target_time, target_time+time_window)

    """ Read in 2km data and merge it as a FeatureGrid """
    data,labels,metas = [],[],{}
    for d,l,m in [abi.get_l2(f) for f in l2_2km_prods]:
        for i in range(len(l)):
            # Skip repeated labels
            if l[i] in labels:
                continue
            labels.append(l[i])
            data.append(d[i])
        # Update the new combined meta-dictionary
        for k in m:
            if k in metas.keys():
                metas[k].append(m[k])
            else:
                metas[k] = [m[k]]

    # Make a FeatureGrid
    fg2k = FeatureGrid(labels=labels, data=data, meta=metas)
    print(fg2k.labels)
    fg2k.get_pixels("norm256 colorize cloud_prob")

    #print(enh.array_stat(fg2k.data("q_acm")))
    #print(fg2k.data("cloud_mask"))
    #masked = fg2k.data("cloud_mask", mask=fg2k.data("cloud_phase")>5)
    #print(masked)
    #gt.quick_render(masked, colorize=True)
    #gt.quick_render(masked, colorize=True, mask=np.ma.getmask(data[1]))
