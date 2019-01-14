from dp.dp import DP

class ContextDP(DP):
    def __init__(self, **kwargs):
        DP.__init__(self, **kwargs)

    def calc(self, jointNames=None, showresult=False, resultdir="", myLocalCosts=None,
             myMatchingCostFunc=None, correspondLine=True, returnMatchingCosts=False):

        pass

    def calc_sync(self, contexts):

        super().calc()