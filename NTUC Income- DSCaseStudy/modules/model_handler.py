from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier 
import pickle
import pandas as pd
import numpy as np

class ModelHandling:
    
    def __init__(self, n_folds=5):
        
        # self.model = []
        self.n_folds = n_folds

        self.base_model1 = GradientBoostingClassifier(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=10,
            max_features='sqrt',
            min_samples_leaf=15,
            min_samples_split=10, 
            random_state =5,
            validation_fraction=0.1,
            n_iter_no_change=50
        )

        self.base_model2 = HistGradientBoostingClassifier(
            loss='binary_crossentropy',
            learning_rate=0.1,
            max_depth=6,
            min_samples_leaf=15,
            random_state =5
        )


        #extracting default parameters from benchmark model
        self.default_params = {}
        gparams = self.base_model1.get_params()

        #default parameters have to be wrapped in lists - even single values - so GridSearchCV can take them as inputs
        for key in gparams.keys():
            gp = gparams[key]
            self.default_params[key] = [gp]

        # self.model = GridSearchCV(estimator=self.base_model1, scoring='roc_auc', param_grid=self.default_params, return_train_score=True, verbose=1, cv=self.n_folds)

    def train_test_split(self, X, y, test_size=0.2, random_state=0):
        X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_tr, X_tt, y_tr, y_tt
    
    def cross_val(self, X, y, scoring_metric):
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=1)
        scores = cross_val_score(self.base_model1, X, y, scoring=scoring_metric, cv=cv, n_jobs=-1)
        return np.mean(scores), scores.min(), scores.max()
    
    def fit(self, X, y):
        self.model = self.base_model1.fit(X, y)
        return self
    
    def feature_importance(self):
        df = pd.DataFrame()
        df['feature'] = self.model.feature_names_in_
        df['feature_importance'] = self.model.feature_importances_
        return df

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def eval_results(self, y, pred, pred_prob):
        """
        calculate different metrics
        """    
        accuracy = accuracy_score(y,pred)
        precision = precision_score(y,pred)
        recall = recall_score(y,pred)
        f1 = f1_score(y,pred)
        roc_auc = roc_auc_score(y, pred_prob)
        return accuracy, precision, recall, f1, roc_auc

    def save_model(self,model,filename):
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self,filename):
        return pickle.load(open(filename, 'rb'))
    
