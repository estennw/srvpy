


class Manifold:
    pass



class PreshapeSpace(Manifold):
    pass






class ScaleSpace(Manifols):

    def __init__(self,scale1,scale2):
        self.scale1 = scale1
        self.scale2 = scale2
        self.log1 = np.log(scale1)
        self.log2 = np.log(scale2)

    def geodesic(self,tau):
        return np.exp(self.log1 + tau*(self.log2-self.log1))
