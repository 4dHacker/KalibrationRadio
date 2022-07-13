import nifty8 as ift
import numpy as np

mat=np.zeros((5,5),dtype=int)


class Op (ift.LinearOperator):
    def apply(self, x, mode):

        if(mode == self.TIMES):

            #schauen, wie viele Antennen und Laden der Entsprechenden Matrix
            with open("AmplitudeMatrix4.npy", 'rb') as f:
                x = np.load(f)
            neheme imaginär teil
            multipliziere mit richtiger Matrix()
            return(closure_phases)
        else:
            print("sas")


operator = Op()
operator.apply(mat,operator.TIMES)