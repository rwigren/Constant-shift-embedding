import numpy as np
import sklearn as skl
import scipy

class CSE(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, dimensions, m_type="sim", symmetrize=True, copy=True):
        '''
        Args:
            dimensions: Amount of dimensions 
            m_type: Type of matrix, similarity (sim) or dissimilarity (dissim)
            copy: If False, data passed to fit are overwritten and running fit(X).transform(X) 
                  will not yield the expected results, use fit_transform(X) instead
        '''
        self.m_type = m_type
        self.dim = dimensions
        self.copy = copy
    
    def fit(self, X, y=None):
        ''' 
        Args: 
            X: Matrix to fit constant shift embedding (CSE) to
        returns:
            Returns the instance itself
        '''
        m_c = self.centralize_matrix(X)
        n, _ = m_c.shape
       
        if self.m_type == "dissim":
            m_c /= -2

        eigs = scipy.linalg.eigh(m_c, eigvals_only=True, eigvals=(0,0))

        diag_ind = np.diag_indices(n)            
        m_c[diag_ind] -= np.amin(eigs)

        eig_range = (n - self.dim, n - 1)            
        eigs, vecs = scipy.linalg.eigh(m_c, eigvals=eig_range)
        self.eigs = eigs[::-1]
        self.vecs = vecs[:, ::-1]
        return self

    def fit_transform(self, X, y=None):
        ''' 
        Args: 
            X: Matrix to fit constant shift embedding (CSE) to
        returns:
            Embeddings 
        '''
        self.fit(X)
        return self.vecs.dot( np.diag( np.sqrt( self.eigs) ) )
    
    def transform(self, X):
        ''' 
        Args: 
            X: Matrix to fit constant shift embedding (CSE) to
        returns:
            Embeddings 
        '''
        m_c = self.centralize_new(X)
        if self.m_type == "dissim":
            m_c /= -2
            
        i_lambda = np.diag( np.sqrt( 1 / self.eigs) )   
        
        return m_c.dot( self.vecs ).dot( i_lambda )   
        
    def centralize_matrix(self, matr):
        ''' Centralizes the matrix and fits for centralization of new matrices
        Args: 
            matr: Matrix to be centralized
        returns:
            centralized matrix 
        '''
        if self.copy:
            A = matr.astype(np.float64)
        else:
            A = matr.view(dtype = np.float64)
            A[:] = matr           
        
        n, _ = A.shape    
        r_mean = np.mean(A, axis = 1) 
        self.c_mean = np.mean(A, axis = 0)
        self.intersect = np.mean(A)

        A -= r_mean[np.newaxis].T
        A -= self.c_mean

        A += self.intersect
        return A
    
    def centralize_new(self, matr):
        ''' Centralizes a new matrix according to previously fit CSE 
        Args: 
            matr: Matrix to be centralized
        returns:
            centralized matrix 
        '''
        if self.copy:
            A = matr.astype(np.float64) 
        else:
            A = matr.view(dtype = np.float64)
            A[:] = matr
        
        r_mean = np.mean(A, axis = 1)
        
        A -= r_mean[np.newaxis].T
        A -= self.c_mean

        A += self.intersect
        return A
    
    def plot_eigs(matr, m_type = "sim"):
        ''' Plots the eigenvalues of the transformed matrix. Useful for determining the dimension of the embeddings
        Args: 
            matr: (N x N) matrix, Similiarity or dissimilarity 
            m_type: Type of matrix, similarity (sim) or dissimilarity (dissim)
        '''
        m_c = CSE.centralize_matrix(matr)
        
        if m_type == "dissim":
            m_c /= -2

        eigs = scipy.linalg.eigh(m_c, eigvals_only=True)
        
        eig_n = np.amin(eigs)
        eigs = eigs - eig_n
        p = np.argsort(eigs)[::-1]

        eigs = eigs[p]
        
        plt.plot(eigs)
        plt.show()
        
        
        
        
        
  