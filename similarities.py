import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy

'''
    Creating a new similarity class:
        ... It is enough to define the single_core. However, any other function, such as matrix_core
        can also be defined for better performance.
'''

''' BASE CLASS '''
class similarity_base():
    fn_type = ''
    @classmethod
    def single(cls, vec1, vec2=None, **kwargs):
        if np.all(vec2 == None):
            return cls.single_fn_self(vec1, **kwargs)
        else:
            return cls.single_fn(vec1, vec2, **kwargs)
    
    @classmethod
    def single_fn(cls, vec1, vec2, **kwargs):
        return 0
    
    @classmethod
    def single_fn_self(cls, vec1, **kwargs):
        return cls.single_fn(vec1, vec1, **kwargs)
    
    @classmethod
    def vector(cls, matr, vec, **kwargs):
        return np.apply_along_axis(cls.single_fn, 1, matr, vec2=vec, **kwargs)
    
    @classmethod
    def matrix(cls, matr1, matr2=None, **kwargs):
        if np.all(matr2 == None):
            return cls.matrix_fn_self(matr1, **kwargs)
        else:
            return cls.matrix_fn(matr1, matr2, **kwargs)
       
    @classmethod
    def matrix_fn(cls, matr1, matr2, **kwargs):
        return np.apply_along_axis(lambda vec, matr: cls.vector(matr, vec), 0, matr2.T, matr=matr1, **kwargs)
    
    @classmethod
    def matrix_fn_self(cls, matr1, **kwargs):
        return cls.matrix_fn(matr1, matr1, **kwargs)
    
    @classmethod
    def auto_sim(cls, e1, e2=None, **kwargs):
        if np.all(e2 == None):
            fn_type = (len(e1.shape), len(e1.shape))
        else:
            fn_type = (len(e1.shape), len(e2.shape))
        
        fn_map = {(1,1): cls.single, 
                  (2,1): cls.vector, 
                  (1,2): lambda vec, matr: cls.vector(matr, vec), 
                  (2,2): cls.matrix}
        return fn_map[fn_type](e1, e2, **kwargs)

''' Similarity classes '''
class cosine_similarity(similarity_base):
    ''' Cosine similarity '''
    fn_type = 'sim'
    
    @classmethod
    def single_fn(cls, vec1, vec2, normalized=False):
        if normalized:
            return vec1.dot(vec2) 
        else:
            return vec1.dot(vec2) / (norm(vec1) * norm(vec2))
    
    @classmethod
    def matrix_fn(cls, matr1, matr2, normalized=False):
        if normalized:
            return matr1.dot(matr2.T)
        else:
            return matr1.dot(matr2.T) / (norm(matr1, axis=1, keepdims=True).dot(norm(matr2, axis=1, keepdims=True).T))
        
    @classmethod
    def vector(cls, matr, vec, normalized=False):
        if normalized:
            return matr.dot(vec)
        else:
            return matr.dot(vec) / (norm(matr, axis=1) * norm(vec))
    
class KLD_2way(similarity_base):
    ''' Two-way Kullback Leibler Divergence '''
    fn_type = 'dissim'
    
    @classmethod
    def single_core(cls, vec1, vec2):
        return entropy(vec1, vec2) + entropy(vec2, vec1)
    
    @classmethod
    def vector(cls, matr, vec):
        mat_log = np.log(matr)
        row_log = np.log(vec)

        kl1 = (row_log- mat_log).dot(vec)
        kl2 = np.sum( np.multiply(mat_log - row_log, matr), axis=1)
        
        return kl1 + kl2
    

class KL_2way():
    ''' A two-way Kullback-Leibler Divergence'''
    fn_type = 'dissim'
    
    @staticmethod
    def core(row, mat):
        mat_log = np.log(mat)
        row_log = np.log(row)

        kl1 = (row_log- mat_log).dot(row)
        kl2 = np.sum( np.multiply(mat_log - row_log, mat), axis=1)
        
        return kl1 + kl2
    
    @staticmethod
    def matrix(mat):
        return np.apply_along_axis(KL_2way.core, 1, mat, mat)

    @staticmethod
    def matrix_new(m1, m2):
        return np.apply_along_axis(KL_2way.core, 1, m2, m1)
    
    @staticmethod
    def vector_new(m, v):
        return KL_2way.core(m, v)
    
    
    
class jaccard_similarity_sparse(similarity_base):
    ''' Jaccard similarity for sparse binary matrices '''
    fn_type = 'sim'
    
    @classmethod
    def single_fn(cls, vec1, vec2):
        res = vec1.dot(vec2)
        return res / (np.sum(vec1) + np.sum(vec2) - np.sum(res))
    
    @classmethod
    def matrix_fn_self(cls, mat):
        '''
            Arguments
                mat: [csr matrix, (n,m)] Sparse Binary Matrix
            Returns
                [ndarray, (n,n)] Dense matrix, in [0, 1]

        '''   
        cols_sum = mat.getnnz(axis=1)
        ab = (mat * mat.T).astype(np.float64)

        aa = np.repeat(cols_sum, ab.getnnz(axis=1)) # for rows    
        bb = cols_sum[ab.indices] # for columns

        ab.data /= (aa + bb - ab.data)
        return ab.toarray()

    @classmethod
    def matrix_fn(cls, matr1, matr2):
        '''
            Arguments
                m1: [csr matrix, (n,m)] Sparse Binary Matrix representing bag of words
                m2: [csr matrix, (k,m)] Sparse Binary Matrix representing bag of words
            Returns
                [ndarray, (k,n)] Dense matrix, in [0, 1]

        '''    
        m1_sums = matr1.getnnz(axis=1) # nx1
        m2_sums = matr2.getnnz(axis=1) # kx1

        ab = (matr2 * matr1.T).astype(np.float64) # - kxn
      
        aa = m1_sums[ab.indices]  # for rows
        bb = np.repeat(m2_sums, ab.getnnz(axis=1))  # for columns

        ab.data /= (aa + bb - ab.data)

        return ab.toarray()
    
    @classmethod
    def vector(cls, matr, vec):
        return cls.matrix_fn(m, v)