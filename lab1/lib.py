from sklearn.base import BaseEstimator, ClassifierMixin


def show_sample(X, index):
	'''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
	
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		index (int): index of digit to show
	'''
	raise NotImplementedError


def plot_digits_samples(X, y):
	'''Takes a dataset and selects one example from each label and plots it in subplots

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	'''
	raise NotImplementedError


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
	'''Calculates the mean for all instances of a specific digit at a pixel location

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select. 

	Returns:
		(float): The mean value of the digits for the specified pixels
	'''
	raise NotImplementedError



def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
	'''Calculates the variance for all instances of a specific digit at a pixel location

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select

	Returns:
		(float): The variance value of the digits for the specified pixels
	'''
	raise NotImplementedError



def digit_mean(X, y, digit):
	'''Calculates the mean for all instances of a specific digit

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select

	Returns:
		(np.ndarray): The mean value of the digits for every pixel
	'''
	raise NotImplementedError



def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
	'''Calculates the variance for all instances of a specific digit

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select

	Returns:
		(np.ndarray): The variance value of the digits for every pixel
	'''
	raise NotImplementedError


def euclidean_distance(s, m):
	'''Calculates the euclidean distance between a sample s and a mean template m

	Args:
		s (np.ndarray): Sample (nfeatures)
		m (np.ndarray): Template (nfeatures)

	Returns:
		(float) The Euclidean distance between s and m
	'''
	raise NotImplementedError


def euclidean_distance_classifier(X, X_mean):
	'''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		X_mean (np.ndarray): Digits data (n_classes x nfeatures)

	Returns:
		(np.ndarray) predictions (nsamples)
	'''
	raise NotImplementedError



class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        Calculates self.X_mean_ based on the mean 
        feature values in X for each class.
        
        self.X_mean_ becomes a numpy.ndarray of shape 
        (n_classes, n_features)
        
        fit always returns self.
        """
        raise NotImplementedError
        #return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        raise NotImplementedError
    
    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        raise NotImplementedError


def evaluate_classifier(clf, X, y, folds=5):
	"""Returns the 5-fold accuracy for classifier clf on X and y

	Args:
		clf (sklearn.base.BaseEstimator): classifier
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)

	Returns:
		(float): The 5-fold classification score (accuracy)
	"""
	raise NotImplementedError