import nifty8 as ift
import numpy as np
import ducc0
import pandas as pd
import math
import matplotlib.pyplot as plt



        
        



def readCSV():
    npArr = []
    for i in range(1,9):
        npArr.append(np.genfromtxt('{}.csv'.format(i), delimiter=','))
    return npArr

def getUVW(nparr):
    uvwArr =[]
    for i in range(8):
        df = pd.DataFrame(nparr[i], columns=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'])
        droplist = ['time', 'T1', 'T2', 'Iamp', 'Iphase', 'Isigma']
        df.drop(droplist, axis=1, inplace=True)
        df['W'] = 0
        uvwArr.append(df.to_numpy())
    return uvwArr

def calcFreq():
    fArr = []
    for i in range(1, 9):
        doc = open('{}.csv'.format(i), 'r')
        line = doc.readline()
        parts = line.split(':')
        fG = parts[3][:8]
        fG = float(fG)
        f=fG*10**9
        fNpArr = np.array([f])  
        fArr.append(fNpArr)
    return fArr

def calcRes(fArr):
    resArr = []
    for i in range(len(fArr)):
        resX = 1.22*((3*(10**8))/(fArr[i][0]*10**7))
        resArr.append(resX)
    return resArr

def calcData(npArr):
    dArr = []
    for i in range(8):
        df = pd.DataFrame(npArr[i], columns=['time', 'T1', 'T2', 'U', 'V', 'Iamp', 'Iphase', 'Isigma'])
        droplist = ['time', 'T1', 'T2','U', 'V', 'Isigma']
        df.drop(droplist, axis=1, inplace=True)
        dn = pd.DataFrame()
        dn['d']=df['Iamp']*math.e**(imaginary*(df['Iphase']))
        dArr.append(dn.to_numpy())
    return dArr

def dirty(uvwArr, dArr, fArr, resArr, pixX, pixY):
    dirtyArr = []
    for i in range(8):
        uvw = uvwArr[i]
        d = dArr[i]
        f = fArr[i]
        res = resArr[i]

        dirty = ducc0.wgridder.experimental.vis2dirty(uvw=uvw, vis=d, freq=f, npix_x=pixX, npix_y=pixY, pixsize_x=res, pixsize_y=res, epsilon=10**-6)
        dirtyArr.append(dirty)
    return dirtyArr

def vis(uvwArr, dArr, fArr, resArr, dirtyArr):
    visArr = []
    for i in range(8):
        uvw = uvwArr[i]
        d = dArr[i]
        f = fArr[i]
        res = resArr[i]
        dirty = dirtyArr[i]

        vis = ducc0.wgridder.experimental.dirty2vis(uvw=uvw, dirty=dirty, freq=f, pixsize_x=res, pixsize_y=res, epsilon=10**-6)
        visArr.append(vis)
    return visArr
    
def plot(dirtyArr):
    '''
    for i in range(8):        
        plt.imshow(dirtyArr[i])
        plt.savefig('auto/autosave{}.png'.format(i))
    '''
    alphas = [1/16+i/16 for i in range(8)]
    print(alphas)
    for i in range(8):        
        plt.imshow(dirtyArr[i], alpha=alphas[i])
    plt.show()

class idk(ift.LinearOperator):
    def __init__(self, domain, target, uvw, freq, res, pixY, pixX):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self.uvw = uvw
        self.freq = freq
        self.res = res
        self.pixX = pixX
        self.pixY = pixY

        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if(mode == self.TIMES):
            dirty = ducc0.wgridder.experimental.vis2dirty(uvw=self.uvw, vis=x.val, freq=self.freq, npix_x=self.pixX, npix_y=self.pixY, pixsize_x=self.res, pixsize_y=self.res, epsilon=10**-6)
            return ift.makeField(self.target, dirty)

        else:
            vis = ducc0.wgridder.experimental.dirty2vis(uvw=self.uvw, dirty=x.val, freq=self.freq, npix_x=self.pixX, npix_y=self.pixY, pixsize_x=self.res, pixsize_y=self.res, epsilon=10**-6)
            return ift.makeField(self.domain, vis)
        
def nifty(uvwArr, dArr, fArr, resArr, pixX, pixY):
    oArr = []
    resArr = []
    for i in range(8):
        print(i)
        rg = ift.RGSpace((pixX, 1))
        field = ift.makeField(rg, dArr[i])
        oArr.append(idk(rg, rg, uvwArr[i], fArr[i], resArr[i], pixX, pixY))
        result = oArr[i](field)
        resArr.append(result)
    return resArr 


imaginary = 0+1j
pixelX = 100
pixelY = 100
def main():

    fArr = calcFreq()
    npArr = readCSV()
    uvwArr = getUVW(npArr)
    
    resArr = calcRes(fArr)
    dArr = calcData(npArr)
    dirtyArr = dirty(uvwArr, dArr, fArr, resArr, pixelX, pixelY)
    #plot(dirtyArr)
    visArr = vis(uvwArr, dArr, fArr, resArr, dirtyArr)
    print(visArr[0])
    #plot(visArr)
    

    print([len(uvwArr), len(fArr), len(dArr), len(resArr)])
    result = nifty(uvwArr, dArr, fArr, resArr, 6458, 1)
    print(result)

main()


#TODO w und uvw, radiar fuer pixsize x und y, vis=daten als nparray d, epsilon 10**-6




