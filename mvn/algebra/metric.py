"""Inspired by https://github.com/pygae/clifford"""
import functools
import itertools
import operator,math

import torch





class CayleyTable:
    
    def __init__(self,n_vectors,grades,num_ud=None):

        basis_dim=sum([math.comb(n_vectors, g) for g in grades])
        self.index_to_bitmap = {}
        self.grades = torch.empty(basis_dim, dtype=int)
        self.bitmap_to_index = {}
#        print("lexi order:",[1 << i for i in range(n_vectors)])
#        print("pwerset",powerset([1 << i for i in range(n_vectors)]))
        
#        for i, t in enumerate(powerset([1 << i for i in range(n_vectors)])):
        if num_ud is not None:
            power_bas = generate_order_powerset(num_ud,grades) #works only for bivectors for now and only one split
        else:
            power_bas = powerset_1([1 << i for i in range(n_vectors)],grades)
        for i, t in enumerate(power_bas):    
#            print (i , t)
            bitmap = functools.reduce(operator.or_, t, 0) #this generates the Clifford basis
#            print("bitmap",bitmap)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t  # enables an optimization inside itertools.combinations





    def construct_gmt_1(self, signature):
        n = len(self.index_to_bitmap)
        array_length = int(n * n)
        k_list = []
        l_list = []
        m_list = []    
        mult_table_vals = []
    # use as small a type as possible to minimize type promotion
   #    mult_table_vals = torch.zeros(array_length)

        for i in range(n):
            bitmap_i = self.index_to_bitmap[i]

            for j in range(n):
            
#            list_ind = i * n + j
            
                bitmap_j = self.index_to_bitmap[j]
                bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            
                if bitmap_v in self.bitmap_to_index:
                    v = self.bitmap_to_index[bitmap_v]
                    k_list.append(i)
                    l_list.append(v)
                    m_list.append(j)
                    mult_table_vals.append(mul)
                    
                    
        coords = torch.tensor([k_list, l_list, m_list])   
        mult_table_vals = torch.tensor(mult_table_vals)     
    
    
#    print("coords",coords, coords.shape)
#    print("mult_table_vals", mult_table_vals, mult_table_vals.shape)
        return torch.sparse_coo_tensor(
            indices=coords, values=mult_table_vals, size=(n, n, n)
            )

def generate_order_powerset(nvec_ud,grade):
    x = [1 << i for i in range(nvec_ud[0])]
    y = [x[-1] << i for i in range(1,nvec_ud[1]+1)]
    x_iter = [itertools.combinations(x, r) for r in grade]
    y_iter = [itertools.combinations(y, r) for r in grade]
    xy = [itertools.product(x, y)]
    
    return itertools.chain.from_iterable(
        x_iter+xy+y_iter[1:])

                 

def powerset_1(iterable,grade):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in grade
    )





def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in `x`, in ascending order"""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


def count_set_bits(bitmap: int) -> int:
    """Counts the number of bits set to 1 in bitmap"""
    count = 0
    for i in set_bit_indices(bitmap):
        count += 1
    return count


def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
#        print("a & bitmap_b", a & bitmap_b)
#        print("count_set_bits",count_set_bits(a & bitmap_b))
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign



