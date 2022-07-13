import nifty8 as ift
import numpy as np
import load_data.main as ld


class Op (ift.LinearOperator):
    def apply(self, x, mode):

        if(mode == self.TIMES):

            #schauen, wie viele Antennen und Laden der Entsprechenden Matrix
            with open("AmplitudeMatrix4.npy", 'rb') as f:
                x = np.load(f)
            #neheme imaginär teil
            #multipliziere mit richtiger Matrix()
            #return(closure_phases)
        else:
            print("sas")




data = ld.load_csv('load_data/1.csv')
antennas = set(data['ant1'].flatten())