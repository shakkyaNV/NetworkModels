from dataclasses import dataclass

X = 15
@dataclass
class HeterodimerModel:
    nodes: int
    k0: float
    k1: float
    k12: float
    k1_tilda: float
    y0_1: list
    y0_2: list
    D: int

    def __post_init__(self):
        assert 0 <= self.k0 <= 1, f"k0 should be between 0 and 1, got {self.k0}"
        assert 0 <= self.k1 <= 1, f"k1 should be between 0 and 1, got {self.k1}"
        assert 0 <= self.k12 <= 1, f"k12 should be between 0 and 1, got {self.k12}"
        assert 0 <= self.k1_tilda <= 1, f"k1_tilda should be between 0 and 1, got {self.k1_tilda}"
        assert self.nodes in [1, 2], f"nodes should be one of [1, 2], got {self.nodes}"
        
        # Returns the appropriate model method based on the number of nodes
        if self.nodes == 1:
            self.modelName = "Heterodimer Single Node Model"
            self.model = self.heterodimer_1_node
            self.y0 = self.y0_1
        elif self.nodes == 2:
            self.modelName = "Heterodimer Two Node Model"
            self.model = self.heterodimer_2_node
            self.y0 = self.y0_2
        else:
            raise ValueError("Currently, only models with 1 or 2 nodes are supported.")

    def heterodimer_1_node(self, t, y):
        """Returning single node heterodimer model."""
        u, v = y

        fu = self.k0 - (self.k1 * u) - (self.k12 * u * v)
        fv = - (self.k1_tilda * v) + (self.k12 * u * v)

        return [fu, fv]

    def heterodimer_2_node(self, t, y):
        """Returning Two Node connected by 1 edge"""

        u_1, v_1, u_2, v_2 = y

        fu_1 =  (-self.D * ( u_1 - u_2)) + self.k0 - (self.k1 * u_1)        - (self.k12 * u_1 * v_1)
        fv_1 =  (-self.D * ( v_1 - v_2))           - (self.k1_tilda * v_1)  + (self.k12 * u_1 * v_1)

        fu_2 =  (-self.D * (-u_1 + u_2)) + self.k0 - (self.k1 * u_2)        - (self.k12 * u_2 * v_2)
        fv_2 =  (-self.D * (-v_1 + v_2))           - (self.k1_tilda * v_2)  + (self.k12 * u_2 * v_2)

        return fu_1, fv_1, fu_2, fv_2


@dataclass
class FisherKolmogorovModel:
    nodes: int
    alpha: float
    y0_1: list
    y0_2: list
    D: int

    def __post_init__(self):
        assert 0 <= self.alpha <= 1, f"alpha should be between 0 and 1, got {self.alpha}"
        assert self.nodes in [1, 2], f"nodes should be one of [1, 2], got {self.nodes}"
        
        # Returns the appropriate model method based on the number of nodes
        if self.nodes == 1:
            self.modelName = "Fisher Kolmogorov Single Node Model"
            self.model = self.fisherKolmogorov_1_node
            self.y0  = self.y0_1
        elif self.nodes == 2:
            self.modelName = "Fisher Kolmogorov Two Node Model"
            self.model = self.fisherKolmogorov_2_node
            self.y0 = self.y0_2
        else:
            raise ValueError("Currently, only models with 1 or 2 nodes are supported.")

    def fisherKolmogorov_1_node(self, t, y):
        c = y

        fc = self.alpha * c * (1 - c)

        return fc

    def fisherKolmogorov_2_node(self, t, y):
        c_1, c_2 = y

        fc_1 = (-self.D * ( c_1 - c_2))  + (self.alpha * c_1 * (1 - c_1))
        fc_2 = (-self.D * (-c_2 + c_2)) + (self.alpha * c_2 * (1 - c_2))

        return fc_1, fc_2