from dp.dp import DP

class HybDP(DP):
    def __init__(self, connections, *args, **kwargs):
        DP.__init__(self, *args, **kwargs)
        self.connections = connections

    def calc(self, jointNames=None, showresult=False, resultdir="",
             myLocalCosts=None, myMatchingCostFunc=None, correspondLine=True, returnMatchingCosts=False):
        print("a")
