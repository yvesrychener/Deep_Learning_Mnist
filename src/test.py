# +-----------------------------------------------------------------------+
# | test.py                                                               |
# | This module implements test of the networks                           |
# +-----------------------------------------------------------------------+

import convnets_siamese
import MLP
import convnet

def run_all():
    '''
    Run all the networks
    '''
    _,_ = MLP.eval_MLP(deep=False)
    _,_ = MLP.eval_MLP(deep=True)
    _,_ = convnet.eval_convnet()
    _,_ = convnets_siamese.eval_transfer_learning()
    _,_ = convnets_siamese.eval_WS_no_AL()
    _,_ = convnets_siamese.eval_WS_AL()
    _,_ = convnets_siamese.eval_no_WS_no_AL()
    _,_ = convnets_siamese.eval_no_WS_AL()
    
if __name__ == "__main__":
    print('Running all the Networks, this may take some time')
    run_all()

