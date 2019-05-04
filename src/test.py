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
   
    # Model 1: shallow MLP
    _,_ = MLP.eval_MLP(deep=False)
    # Model 2: deep MLP
    _,_ = MLP.eval_MLP(deep=True)
    # Model 3: naive ConvNet
    _,_ = convnet.eval_convnet()
    # Model 4: refined ConvNet with no WS & no AL (baseline)
    _,_ = convnets_siamese.eval_no_WS_no_AL()
    # Model 5: refined ConvNet with WS
    _,_ = convnets_siamese.eval_WS_no_AL()
    # Model 6: refined ConvNet with AL
    _,_ = convnets_siamese.eval_no_WS_AL()
    # Model 7: refined ConvNet with WS & AL
    _,_ = convnets_siamese.eval_WS_AL()
    # Model 8: transfer learning
    _,_ = convnets_siamese.eval_transfer_learning()

    
if __name__ == "__main__":
    print('Running all the networks, this may take some time')
    run_all()

