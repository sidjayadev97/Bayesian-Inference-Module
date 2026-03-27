import umap
import sklearn
from sklearn.model_selection import train_test_split
from auxiliary_functions import *


class FeatureExtractor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.features_extracted = None
        self.feature_names = None
        
    def train_test_split(self, train_percent=70):
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.X, self.y, 
                                                                             train_size=(train_percent/100), random_state=2)
    def get_features(self, num_features):
        if is_tabular(self.X_train):
            # if data is tabular
            # we can use random forest feature_importance method
            if ( numpy.isnan(self.X_train).any() or numpy.isnan(self.y_train).any() ):
                # first check if any missing values
                # so if either condition is true, there are missing values
                return 'Your data-matrix or target-class has missing values. Please fill them in first.'
            else:
                # data is all complete, do mutual information feature extractor
                # selectKbest
                KBest = SelectKBest(mutual_info_classif, _?__)
                transformed = KBest.fit_transform(self.X_train, self.y_train)
                self.features_extracted = transformed
                print('{} best features extracted!'.format(num_features))
        elif is_image(self.X_train):
            # we can only use UMAP
            # we need to first reshape self.X_train from 3D numpy array to 2D array
            X_train_reshaped = np.reshape(self.X_train, (len(self.X_train), -1))
            # then use UMAP
            mapper = umap.UMAP(n_components=num_features, random_state=2)
            mapper.fit(X_train_reshaped)
            umap_features = mapper.transform(X_train_reshaped)
            self.features_extracted = umap_features
            print('{} best features extracted!'.format(num_features))