import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nifty8 as ift
import sys


def getFreq(filename):
    with open(filename, 'r') as doc:
        line = doc.readline()
        parts = line.split(':')
        fG = float(parts[3][:8])
    return fG*10**9


def getAnts(filename):
    df = pd.read_csv(filename,  names=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'], header=1)
    droplist = ['time', 'U', 'V', 'Iamp', 'Iphase', 'Isigma']
    df.drop(droplist, axis=1, inplace=True)
    return df.to_numpy()


def getUvw(filename):
    df = pd.read_csv(filename, names=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'], header=1)
    droplist = ['time', 'T1', 'T2', 'Iamp', 'Iphase', 'Isigma']
    df.drop(droplist, axis=1, inplace=True)
    df['W'] = 0
    return df.to_numpy()


def getVis(filename):
    df = pd.read_csv(filename, names=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'], header=1)
    droplist = ['time', 'T1', 'T2','U', 'V', 'Isigma']
    df.drop(droplist, axis=1, inplace=True)
    dn = pd.DataFrame()
    amp = df["Iamp"].to_numpy()
    ph = df["Iphase"].to_numpy()
    dn['d']=amp*np.exp(1j*ph/180*np.pi)
    return dn.to_numpy()


def getTime(filename):
    df = pd.read_csv(filename, names=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'], header=1)
    droplist = ['T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma']
    df.drop(droplist, axis=1, inplace=True)
    return np.squeeze(df.to_numpy())


def load_data(day, freqId):
    if day not in ["095", "096", "100", "101"]:
        raise ValueError("Day can only be 095, 096, 100, 101")
    if freqId not in ["hi", "lo"]:
        raise ValueError()

    freqId = "hi"
    data_file = f'data/SR1_M87_2017_{day}_{freqId}_hops_netcal_StokesI.csv'
    time = getTime(data_file)
    ant1, ant2 = getAnts(data_file).T
    uvw = getUvw(data_file)
    freq = getFreq(data_file)
    vis = getVis(data_file)
    d_space = ift.UnstructuredDomain(vis.shape)
    vis = ift.makeField(d_space, vis)
    d = {"uvw": uvw, "freq": freq, "vis": vis, "time": time, "ant1": ant1, "ant2": ant2}
    preprocess_data(d)
    return d


def preprocess_data(d):
    unique_station_names = set(list(d["ant1"])).union(set(list(d["ant2"])))
    dct = {kk: ii for ii, kk in enumerate(unique_station_names)}
    new_ant1 = np.array([dct[aa] for aa in d["ant1"]])
    new_ant2 = np.array([dct[aa] for aa in d["ant2"]])

    must_be_fixed = new_ant1 >= new_ant2

    final_ant1 = new_ant1.copy()
    final_ant2 = new_ant2.copy()

    final_ant1[must_be_fixed] = new_ant2[must_be_fixed]
    final_ant2[must_be_fixed] = new_ant1[must_be_fixed]

    new_vis = d["vis"].val.copy()
    new_vis[must_be_fixed] = new_vis[must_be_fixed].conjugate()
    d["vis"] = ift.makeField(d["vis"].domain, new_vis)

    d["ant1"] = final_ant1
    d["ant2"] = final_ant2

    assert np.all(d["ant1"] < d["ant2"])


class RadioResponse(ift.LinearOperator):
    def __init__(self, domain, uvw, freq, epsilon):
        self._domain = ift.DomainTuple.make(domain)
        target = ift.UnstructuredDomain((uvw.shape[0], 1))
        self._target = ift.DomainTuple.make(target)
        self.uvw = uvw
        self.freq = np.array([freq])
        self.pixX, self.pixY = domain.shape
        self.dx, self.dy = self._domain[0].distances
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._eps = float(epsilon)

    def apply(self, x, mode):
        from ducc0.wgridder.experimental import dirty2vis, vis2dirty
        self._check_input(x, mode)
        if mode == self.TIMES:
            vis = dirty2vis(uvw=self.uvw, dirty=x.val, freq=self.freq,
                            npix_x=self.pixX, npix_y=self.pixY,
                            pixsize_x=self.dx, pixsize_y=self.dy,
                            epsilon=self._eps)
            return ift.makeField(self.target, vis)
        else:
            dirty = vis2dirty(uvw=self.uvw, vis=x.val, freq=self.freq,
                              npix_x=self.pixX, npix_y=self.pixY,
                              pixsize_x=self.dx, pixsize_y=self.dy,
                              epsilon=self._eps)
            return ift.makeField(self.domain, dirty)


def get_Response(s_space, d, epsilon=1e-6):
    return RadioResponse(s_space, d["uvw"], d["freq"], epsilon)


def main(day):
    MUAS2RAD = 1/1e6/3600 * np.pi/180
    nx, ny = 100, 100
    fovx, fovy = 200*MUAS2RAD, 200*MUAS2RAD
    dx, dy = fovx / nx, fovy / ny

    s_space = ift.RGSpace((nx, ny), (dx, dy))
    d = load_data(day, "hi")

    # MODELL-DEFINITION
    # Sky-modell: f: xi -> sky
    # FIXME

    # sky -> visibilities
    R = get_Response(s_space, d)
    ift.single_plot(R.adjoint(d["vis"]), name="Rdagger_d.png")

    # visibilities -> closure phases
    import closure_stuff
    clph = closure_stuff.getClosurePhaseOperator(d)
    exit()
    # FIXME

    # closure phase likelihood
    # FIXME

    # /MODELL-DEFINITION


    # Bayes: Berechne P(xi | d)
    # FIXME



if __name__ == "__main__":
    _, day = sys.argv
    main(day)
