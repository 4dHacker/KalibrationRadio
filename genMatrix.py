import math
import pandas as pd
import numpy as np
import pickle
import nifty8 as ift
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

n=4
np.set_printoptions(threshold=np.inf)
def genPhaseMatrix(n):
    O=math.comb(n,2)
    mat=np.zeros(O,dtype=int)
    pos=[(x,y) for x in range(n) for y in range(x+1,n)]
    maxRank=math.comb(n-1,2)
    for i in range(0,n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                row=np.zeros(O,dtype=int)
                row[pos.index((i,j))]=1
                row[pos.index((i,k))]=-1
                row[pos.index((j,k))]=1

                newMat=np.vstack([mat,row])
                rnk= np.linalg.matrix_rank(newMat)
                if rnk>np.linalg.matrix_rank(mat):
                    mat=newMat
                    if rnk==maxRank:
                        return mat[1:,:]
    mat=mat[1:,:]
    dom=ift.UnstructuredDomain(mat.shape)
    fied=ift.Field(dom,mat)
    return fied

def genAmpMatrix(n):
    O = math.comb(n, 2)
    mat = np.zeros(O, dtype=int)
    pos = [(x, y) for x in range(n) for y in range(x + 1, n)]#scipi sparsematrix
    maxRank = math.comb(n - 1, 2)
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                for l in range(k+1,n):
                    row = np.zeros(O, dtype=int)
                    row[pos.index((i,j))]=1
                    row[pos.index((k,l))]=1
                    row[pos.index((i,k))]=-1
                    row[pos.index((j,l))]=-1

                    newMat = np.vstack([mat, row])
                    rnk = np.linalg.matrix_rank(newMat)
                    if rnk > np.linalg.matrix_rank(mat):
                        mat = newMat
                        #if rnk == maxRank:
                            #return mat[1:, :]

                    row = np.zeros(O, dtype=int)
                    row[pos.index((i, l))] = 1
                    row[pos.index((j, k))] = 1
                    row[pos.index((i, k))] = -1
                    row[pos.index((j, l))] = -1

                    newMat = np.vstack([mat, row])
                    rnk = np.linalg.matrix_rank(newMat)
                    if rnk > np.linalg.matrix_rank(mat):
                        mat = newMat
    return mat[1:,:]

for i in range(3,36):
    print(i)
    fld=genPhaseMatrix(i)
    filehand=open(f'{i}-PhaseMatrix.obj','wb')
    pickle.dump(fld,filehand)
print('test')



