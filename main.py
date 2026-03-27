from auxiliary_functions import *
from features_tech import *
from model_tech import *

class BayesClassifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        image_data = None
        tabular_data = None
        self.categories = np.unique(y)
        self.log_priors = None
        self.log_likelihoods = None
        self.log_posteriors = None
        self.X_features = None
        self.model_dict = None
        if is_image(X):
            image_data = True
        elif is_tabular(X):
            tabular_data = True
        else:
            return 'Please load in either image or tabular data.'
      
    
    def preprocess(self, n_features):
        Fext = FeatureExtractor(self.X, self.y)
        Fext.train_test_split()
        Fext.get_features(num_features=n_features)
        self.X_features = Fext.features_extracted
        self.y_train, self.y_test, self.X_test = Fext.y_train, Fext.y_test, Fext.X_test  
            
    def use_model(self, model_type):
        # model_type is the type of model. KernelDensity or GaussianMixture (from sklearn) work as values
        if ( model_type is sklearn.neighbors.KernelDensity ):
            Model = LikelihoodModel(self.X_features, self.y_train, sklearn.neighbors.KernelDensity)
            Model.fit()
            self.model_dict = Model.model_dict
        elif ( model_type is sklearn.mixture.GaussianMixture ):
            Model = LikelihoodModel(self.X_features, self.y_train, sklearn.mixture.GaussianMixture)
            Model.fit()
            self.model_dict = Model.model_dict
    
    def predict_proba(self):
        # this function will do the actual posterior probability  prediction 
        # we add the log_priors + log_likelihoods = log_posteriors (and normalize) 
        if image_data:
            X_test_reshaped = np.reshape(self.X_test, (len(self.X_test), -1))
            # we need to first reshape self.X_train from 3D numpy array to 2D array
            # for each row (data-point) in the test data-matrix
            self.log_likelihoods = np.asarray([model.score(X_test_reshaped) for model in self.model_dict])
            self.log_priors = np.log(pd.Series(self.y_train).value_counts(normalize=True).values)
            norm = np.add(self.log_likelihoods, self.log_priors)
            self.posteriors = norm/np.sum(norm)     
        elif tabular_data:
            # we do NOT need to reshape anything since its already like a table
            self.log_likelihoods = np.asarray([model.score(X_test_reshaped) for model in self.model_dict])
            self.log_priors = np.log(np.asarray(pd.Series(self.y_train).value_counts(normalize=True).values))
            normalized = np.add(self.log_likelihoods, self.log_priors)
            # normalized is a np-array
            self.log_posteriors = normalized/np.sum(normalized)
                                     
    def predict_class(self):
        # we'll use the results of the predict_proba() function 
        # self.log_posteriors contains the Logarithm(posterior probabilities) already normalized!
        # so we just pick the index (in the np-array) corresponding to highest posterior, report that
        # (along with its % probability)
        posterior_copy = 0 
        index_firstplace = self.log_posteriors.index(max(self.log_posteriors))
        index_secondplace = 0
        

    def main(self, X, y):
        BayesObj = BayesClassifier(X, y) 



    
  

if name == "__main__":
    main()
    