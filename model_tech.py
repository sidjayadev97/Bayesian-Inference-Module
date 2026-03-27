from auxiliary_functions import *
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


class LikelihoodModel:
    def __init__(self, X, y, model_type):
        self.X = X
        self.y = y
        self.categories = np.unique(y)
        self.model_dict = None 
        self.model_type = model_type # the model inputted here 
    
    def fit(self):
        category_models = []
        for category in self.categories:
            category_index = np.where( self.y==category )
            X_category = self.X[category_index]
            # get input about parameter grid used on model
            # now check for the model type
            if ( self.model_type is sklearn.neighbors.KernelDensity ):
                # fill in parameter dictionary
                key = input('Enter parameter name: ')
                start = float(input('Enter start value: '))
                end = float(input('Enter end value: '))
                step = float(input('# of points? '))
                param_grid = {key: np.logspace(start, end, step)}
                # grid-search
                optimal_model = find_best_params( X_category, KernelDensity(kernel='gaussian'), param_grid )
                category_models.append( optimal_model )
            # otherwise if the model is Gaussian Mixture 
            elif ( self.model_type is sklearn.mixture.GaussianMixture ):
                # fill in parameter dictionary
                key = input('Enter parameter name: ')
                n_components = int(input('# of components? ')) 
                param_grid = {key: np.asarray(list(range(1, n_components+1))) }
                # grid-search
                optimal_model = find_best_params( X_category, GaussianMixture(n_components=n_components), param_grid )
                category_models.append( optimal_model )    
        # put all the models for each category into the model_dict instance variable        
        self.model_dict = {('Model {i}'.format(i)):m for (i,m) in enumerate(category_models)}
        
    def visualize(self):
        # first, choose a model from the self.model_dict
        
        pass