import functools
import math

import torch
from torch import nn

from .metric import  gmt_element, CayleyTable


class CliffordAlgebra(nn.Module):
    def __init__(self, metric ,split_metric=None, input_grades=None):
        super().__init__()
    
        self.register_buffer("metric", torch.as_tensor(metric))
        self.num_bases = len(metric)
#        print("num_bases:",self.num_bases)
        self.dim = len(self.metric)
#        print("dim",self.dim)
#        self.input_grades = [0,1,2,3] #user input
        self.input_grades = input_grades 
        self.n_subgrade = len(self.input_grades)
        self.bbo = CayleyTable(self.num_bases,self.input_grades,num_ud=split_metric)
        self.n_blades = len(self.bbo.grades)

#        self.n_blades = 2**self.num_bases
#        print("n_blades",self.n_blades)
        
        self.grades = self.bbo.grades.unique().tolist()      

        cayley = (
            (self.bbo.construct_gmt_1(self.metric)).to_dense().to(torch.get_default_dtype()))
        
        

        
        #all the info below can be obtained from the Cayley class
 #       self.grades = self.bbo.grades.unique()
#        print("gardes",self.grades)
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
#        print("n_subspaces",self.n_subspaces)
#        print("subspace",self.subspaces)
#        grade_to_slice_1 = self._grade_to_slice(self.subspaces)
        self.grade_to_slice = self._grade_to_slice_sub() #its a dict now
#        grade_to_slice_2_converted = {k.item(): v for k, v in grade_to_slice_2.items()}
#        lst = [grade_to_slice_2_converted[i] for i in range(len(grade_to_slice_2_converted))]
        

        grade_to_index_list = [
            torch.tensor(range(*s.indices(s.stop))) for s in list(self.grade_to_slice.values())
        ]
        self.grade_to_index = dict(zip(self.grade_to_slice.keys(), grade_to_index_list)) #its a dict now

#        print("grade_to_slice", self.grade_to_slice)
#        print("grade_to_index", self.grade_to_index)        
        
        self.register_buffer(
            "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
#        print("even grades:", self.even_grades)
#        print("odd grades:", self.odd_grades)
        self.register_buffer("cayley", cayley)

    def geometric_product(self, a, b, blades=None):
        cayley = self.cayley

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
 #           print("blades_l, blades_o and blade_r", blades_l, blades_o,blades_r)
 #           print("shapes of blades_l, blades_o and blade_r", blades_l.shape, blades_o.shape,blades_r.shape)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)

    def _grade_to_slice(self, subspaces):
#        print(" i a m here")
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    def _grade_to_slice_sub(self): #this just concatenates the different grades, when working in a subspace of the full space
#        print(" i a m here")
        grade_to_slice = {}
#        subspaces = torch.as_tensor(subspaces)
        index_start = 0
        for grade in self.grades:
 #           index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade) 
            grade_to_slice[grade] = slice(index_start, index_end)
            index_start = index_end 
        return grade_to_slice
    
    @functools.cached_property
    def _alpha_signs(self):
        return torch.pow(-1, self.bbo_grades)

    @functools.cached_property
    def _beta_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2)

    @functools.cached_property
    def _gamma_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2)

    def alpha(self, mv, blades=None):
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], 2**self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        return mv

    def embed_sub(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], self.n_blades, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        return mv


    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], 2**self.dim, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def embed_grade_sub(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], self.n_blades, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        return mv[..., s]

#=============================================================================
    def get_grade_sub(self, mv: torch.Tensor, grade: int, subsubspace) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        s0 = mv[...,0]
        temp = mv[..., s]
        return s0, temp[...,subsubspace[0]:subsubspace[1]]

#=============================================================================

    def b(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
#        print("shape from q", self.b(mv, mv, blades=blades).shape)    
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def norm(self, mv, blades=None):
#        print("shape of norm",self._smooth_abs_sqrt(self.q(mv, blades=blades)).shape)
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

 
    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

 
    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def sandwich(self, u, v, w):
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)

        return torch.tensor(blades)

    def random(self, n=None):
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)

        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def rho(self, w, mv):
        """Applies the versor w action to mv."""
        return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths

    @functools.cached_property
    def geometric_product_paths_sub(self):
        gradedim = len(self.grades)
        gp_paths = torch.zeros((gradedim, gradedim, gradedim), dtype=bool)

        for i in self.grades:
            for j in self.grades:
                for k in self.grades:
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths
    
    def split(self, mv, lastdim=None):
#        num_bases = 2 ** self.dim
        batch_size = mv.shape[0]
        if lastdim is not None:
            return mv.reshape(batch_size, -1, lastdim)
        else:    
            return mv.reshape(batch_size, -1, self.n_blades)

    def flatten(self, mv):
        batch_size = mv.shape[0]
        return mv.reshape(batch_size, -1)    