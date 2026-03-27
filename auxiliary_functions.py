from sklearn.model_selection import GridSearchCV

def find_best_params(X, model_inst, param_grid):
    if ( isinstance(model_inst, sklearn.neighbors.KernelDensity) ):
        grid = GridSearchCV( estimator=model_inst, 
                                param_grid=param_grid, n_jobs=-1 )
        grid.fit(X)
        best_model = sklearn.neighbors.KernelDensity(kernel='gaussian', grid.best_params[list(param_grid)[0]])
        best_model.fit(X)
    elif ( isinstance(model_inst, sklearn.mixture.GaussianMixture) ):
        grid = GridSearchCV( estimator=model_inst, 
                                param_grid=param_grid, n_jobs=-1 )
        grid.fit(X)
        best_model = sklearn.mixture.GaussianMixture(n_components=grid.best_params[list(param_grid)[0]])
        best_model.fit(X)
    else:
        return 'Error handling user-inputted model instance name.'
    return best_model


def is_tabular(X):
    # check the dimensions of the data-matrix
    # if it is 2-dimensional numpy array, its tabular
    if (X.ndim == 2):
        return True
    else:
        return False
  
    
def is_image(X):
    # check the dimensions of the data-matrix
    # if it is 3-dimensional numpy array, its image
    if (X.ndim == 3):
        return True
    else:
        return False 