import numpy as np
import os
from racetrack_library import load_racing_line_modified
def racetrack_feature_pre_extracted(tracks_dir,racing_line_dir,filename,with_dists):
    track_file=os.path.join(tracks_dir,filename)
    rcl=os.path.join(racing_line_dir,filename)
    with open(track_file, "r") as f:
        lines = f.readlines()

    l,alpha,thetas,distances=[],[],[],[]
    racingline=load_racing_line_modified(rcl)
    racingline=np.array(racingline,np.float64)
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 4 and "#"!=line[0]:
            tl,talpha,tthetas,tdistance = map(np.float64,parts)
            l.append(tl)
            alpha.append(talpha)
            thetas.append(tthetas)
            distances.append(tdistance)
    if with_dists==1:
        return l,alpha,thetas,distances,racingline
    else:
        return l,alpha,thetas,racingline