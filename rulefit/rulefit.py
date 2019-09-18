"""Linear model of tree-based decision rules

This method implement the RuleFit algorithm

The module structure is the following:

- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm

"""
import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from functools import reduce


__all__ = ['RuleFit']



class Winsorizer():
    """Performs Winsorization 1->1*
    Warning: this class should not be used directly.
    """    
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.winsor_lims=None
        
    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]
        
    def trim(self,X):
        X_=X.copy()
        X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
        return X_


class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

    Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """    
    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.scale_multipliers = None
        self.winsor_lims = None
        
    def train(self, X):
        # get winsor limits
        self.winsor_lims = np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]
        # get multipliers
        X_trimmed=self.trim(X)
        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(X_trimmed[:,i_col]))
        self.scale_multipliers=scale_multipliers
        
    def scale(self,X,winsorize=True):
        if winsorize:
            return self.trim(X)*self.scale_multipliers
        else:
            return X*self.scale_multipliers
        
    def trim(self,X):
        X_=X.copy()
        X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
        return X_



class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 na_direction=None,
                 feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.na_direction = na_direction
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X: pd.Series):
        if self.operator == "<=":
            res = 1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        elif self.operator == '==':
            res = 1 * np.array([i in self.threshold for i in X[:, self.feature_index]])
        elif self.operator == '!=':
            res = 1 * np.array([i not in self.threshold for i in X[:, self.feature_index]])
        else:
            raise ValueError('{} is not a valid operator'.format(self.operator))

        if self.na_direction is not None:
            na_ind = np.isnan(X[:, self.feature_index])
            res[na_ind] = int(self.na_direction == 'left')
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, 
                    tuple(self.threshold) if isinstance(self.threshold, list) else self.threshold, 
                    self.operator, 
                    self.feature_name))



class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self, rule_conditions):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        # self.prediction_value=prediction_value
        # self.rule_direction=None

    def transform(self, X: pd.Series):
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __len__(self):
        return len(self.conditions)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # TODO: sum of hash values looks kinda sketchy
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()




_OPERATORS = {
    '<=': '>',
    '>': '<=',
    '!=': '==',
    '==': '!='
}

def get_opposite_operator(op):
    return _OPERATORS[op]



def extract_rules_from_scikit_tree(tree, feature_names=None):
    """ Extract a scikit-learn DecisionTree object into a set of Rule"""
    if hasattr(tree, 'tree_'):
        tree = tree.tree_

    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []

        ## if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]: 
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            
            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else: # a leaf node
            if len(new_conditions)>0:
                new_rule = Rule(new_conditions)
                rules.update([new_rule])
            else:
                pass #tree only has a root node!
            return None

    traverse_nodes()
    return rules


def extract_rules_from_lgbm_tree(tree: dict, feature_names=None):
    """ Extract a set of Rule from a tree from a lightgbm booster
        the tree is one of the tree object from booster.dump_model()['tree_info']
    """
    if 'tree_structure' in tree:
        tree = tree['tree_structure']

    rules = set()
    n_total_sample = tree['internal_count']
    
    def traverse_nodes(tree,
                       operator=None,
                       threshold=None,
                       feature=None,
                       n_sample=None,
                       na_direction='left',
                       conditions=[]):
        if tree.get('split_index', None) == 0:
            new_conditions = []
        else:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
                
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=n_sample / n_total_sample,
                                           na_direction=na_direction,
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
        
        ## if not terminal node
        if 'leaf_index' not in tree: 
            feature = tree['split_feature']
            threshold = tree['threshold']
            operator = tree['decision_type']
            if operator == '==':
                threshold = [float(i.strip()) for i in threshold.split('||')]
            else:
                threshold = float(threshold)
            n_sample = tree['internal_count']
            na_direction = 'left' if tree['default_left'] else 'right'
            
            traverse_nodes(tree=tree['left_child'], 
                           operator=operator,
                           threshold=threshold,
                           feature=feature,
                           n_sample=n_sample,
                           na_direction=na_direction,
                           conditions=new_conditions)
            
            traverse_nodes(tree=tree['right_child'], 
                           operator=get_opposite_operator(operator),
                           threshold=threshold,
                           feature=feature,
                           n_sample=n_sample,
                           na_direction=na_direction,
                           conditions=new_conditions)
            
        else: # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions)
                rules.update([new_rule])
            else:
                pass #tree only has a root node!
            return None

    traverse_nodes(tree)
    return rules


class RuleEnsemble():
    """Ensemble of binary decision rules

    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.

    Parameters
    ----------
    model: List of DecisionTree, a random forest or a booster

    feature_names: List of strings, optional (default=None)
        Names of the features

    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """
    def __init__(self,
                 model,
                 model_type='lightgbm',
                 feature_names=None):
        if model_type not in ('tree', 'forest', 'gbdt', 'lightgbm'):
            raise ValueError('Only supported model types are: {}'.format(['tree', 'forest', 'gbdt', 'lightgbm']))

        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules= list(self.rules)

    def _extract_rules(self):
        """ Recursively extract rules from the model """
        if self.model_type == 'tree':
            # a list of decision tree
            for tree in self.model:
                rules = extract_rules_from_scikit_tree(tree, feature_names=self.feature_names)
                self.rules.update(rules)
        
        elif self.model_type in ('forest', 'gbdt'):
            for tree in self.model.estimators_:
                rules = extract_rules_from_scikit_tree(tree, feature_names=self.feature_names)
                self.rules.update(rules)
        
        elif self.model_type == 'lightgbm':
            if hasattr(self.model, 'booster_'):
                model = self.model.booster_
                for tree in model.dump_model()['tree_info']:
                    rules = extract_rules_from_lgbm_tree(tree, feature_names=self.feature_names)
                    self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = list(filter(lambda x: func(x), self.rules))

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x) > k)

    def transform(self, X, coefs=None):
        """Transform dataset.

        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        coefs:  (optional) if supplied, this makes the prediction
                slightly more efficient by setting rules with zero 
                coefficients to zero without calling Rule.transform().
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list = self.rules 
        if coefs is None :
            return np.array([rule.transform(X) for rule in rule_list]).T
        else: # else use the coefs to filter the rules we bother to interpret
            res= np.array([rule_list[i_rule].transform(X) \
                        for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T
            res_=np.zeros([X.shape[0],len(rule_list)])
            res_[:,coefs!=0]=res
            return res_

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()



class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True, 
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree. 
                        FP 2004 (Sec. 2)
        max_rules:      approximate total number of rules generated for fitting. Note that actual
                        number of rules will usually be lower than this due to duplicates.
        memory_par:     scale multiplier (shrinkage factor) applied to each new tree when 
                        sequentially induced. FP 2004 (Sec. 2)
        rfmode:         'regress' for regression or 'classify' for binary classification.
        lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                        by multiplying the winsorised variable by 0.4/stdev.
        lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear 
                        terms before standardisation.
        exp_rand_tree_size: If True, each boosted tree will have a different maximum number of 
                        terminal nodes based on an exponential distribution about tree_size. 
                        (Friedman Sec 3.3)
        model_type:     'r': rules only; 'l': linear terms only; 'rl': both rules and linear terms
        random_state:   Integer to initialise random objects and provide repeatability.
        tree_generator: Optional: this object will be used as provided to generate the rules. 
                        This will override almost all the other properties above. 
                        Must be GradientBoostingRegressor or GradientBoostingClassifier, optional (default=None)

    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble

    feature_names: list of strings, optional (default=None)
        The names of the features (columns)

    """
    def __init__(self,
                tree_size=4,
                categorical_cols=None,
                sample_fract=1, 
                max_rules=2000,
                learning_rate=0.01,
                model_type='lightgbm',
                mode='classification',
                lin_trim_quantile=0.025,
                lin_standardise=True, 
                exp_rand_tree_size=True,
                include_linear_features=True,
                Cs=None,
                cv=3,
                random_state=1024):
        if model_type not in ('tree', 'forest', 'gbdt', 'lightgbm'):
            raise ValueError('Supported model types are: {}.'.format(['tree', 'forest', 'lightgbm']))

        if mode not in ('classification', 'regression'):
            raise ValueError('Mode should be one of classification and regression.')

        if mode == 'lightgbm' and exp_rand_tree_size:
            warnings.warn('Randomized tree depth is not supported when using LGBM as rule generator. '
                          'Using constant depth instead.')

        self.tree_size = tree_size
        self.categorical_cols = categorical_cols
        self.sample_fract = sample_fract 
        self.max_rules = max_rules
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.mode = mode
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.friedscale = FriedScale(trim_quantile=lin_trim_quantile)
        self.exp_rand_tree_size = exp_rand_tree_size
        self.include_linear_features = include_linear_features
        self.cv = cv
        self.Cs = Cs
        self.random_state = random_state

    def _fit_scikit_estimator_with_warm_start(self, X, y, clf):
        """ Used to train RF and GBDT """
        N = X.shape[0]

        if not self.exp_rand_tree_size: 
            # simply fit with constant tree size
            clf.fit(X, y)
        else: 
            # randomise tree size as per Friedman 2005 Sec 3.3
            np.random.seed(self.random_state)
            tree_sizes=np.random.exponential(scale=self.tree_size-2,
                                            size=int(np.ceil(self.max_rules*2/self.tree_size)))
            tree_sizes=np.asarray([2+np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],dtype=int)
            i=int(len(tree_sizes)/4)
            while np.sum(tree_sizes[0:i])<self.max_rules:
                i=i+1
            tree_sizes=tree_sizes[0:i]
            clf.set_params(warm_start=True) 
            curr_est_=0
            for i_size in np.arange(len(tree_sizes)):
                size=tree_sizes[i_size]

                # warm_state=True seems to reset random_state, such that the trees are highly correlated, 
                # unless we manually change the random_sate here.
                random_state = self.random_state if self.random_state else 0
                random_state += i_size

                clf.set_params(**{'n_estimators': curr_est_+1,
                                'max_leaf_nodes': size,
                                'random_state': random_state})
                clf.fit(X, y)

                curr_est_=curr_est_+1
            clf.set_params(warm_start=False) 

        return clf

    def _fit_random_forest(self, X, y):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        N = X.shape[0]
        n_estimators = self.max_rules // self.tree_size
        sample_fract = self.sample_fract or min(0.5, (100+6*np.sqrt(N))/N)

        if self.mode == 'classification':
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         max_leaf_nodes=self.tree_size,
                                         subsample=sample_fract,
                                         max_depth=100,
                                         random_state=self.random_state)
        else:
            clf = RandomForestRegressor(n_estimators=n_estimators,
                                         max_leaf_nodes=self.tree_size,
                                         subsample=sample_fract,
                                         max_depth=100,
                                         random_state=self.random_state)
        
        return self._fit_scikit_estimator_with_warm_start(X, y, clf)

    def _fit_gbdt(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        N = X.shape[0]
        n_estimators = self.max_rules // self.tree_size
        sample_fract = self.sample_fract or min(0.8, (100+6*np.sqrt(N))/N)

        if self.mode == 'classification':
            clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                             learning_rate=self.learning_rate,
                                             max_leaf_nodes=self.tree_size,
                                             subsample=sample_fract,
                                             max_depth=100,
                                             random_state=self.random_state)
        else:
            clf = GradientBoostingRegressor(n_estimators=n_estimators,
                                            learning_rate=self.learning_rate,
                                            max_leaf_nodes=self.tree_size,
                                            subsample=sample_fract,
                                            max_depth=100,
                                            random_state=self.random_state)
        
        return self._fit_scikit_estimator_with_warm_start(X, y, clf)


    def _fit_lightgbm(self, X, y):
        from lightgbm import LGBMClassifier, LGBMRegressor

        N = X.shape[0]
        n_estimators = self.max_rules // self.tree_size
        sample_fract = self.sample_fract or min(0.8, (100+6*np.sqrt(N))/N)

        if self.mode == 'classification':
            model = LGBMClassifier(num_leaves=self.tree_size,
                                   learning_rate=self.learning_rate,
                                   subsample=sample_fract,
                                   n_estimators=n_estimators,
                                   zero_as_missing=False)
        else:
            model = LGBMRegressor(num_leaves=self.tree_size,
                                  learning_rate=self.learning_rate,
                                  subsample=sample_fract,
                                  n_estimators=n_estimators,
                                  zero_as_missing=False)
        model.fit(X, y, feature_name=self.feature_names, categorical_feature=self.categorical_cols)
        return model

    def fit(self, X, y=None, feature_names=None):
        """Fit and estimate linear combination of rule ensemble

        """
        # Enumerate features if feature names not provided
        N = X.shape[0]

        if hasattr(X, 'columns'):
            # X is a DataFrame
            feature_names = X.columns.tolist()
            X = X.copy().values

        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # build trees
        if self.model_type == 'forest':
            model = self._fit_random_forest(X, y)
        elif self.model_type == 'tree':
            # build a list of decision trees by training a RF
            model = self._fit_random_forest(X, y)
            model = model.estimators_
        elif self.model_type == 'gbdt':
            model = self._fit_gbdt(X, y)
        elif self.model_type == 'lightgbm':
            model = self._fit_lightgbm(X, y)
        self.model = model 

        # extract rules
        self.rule_ensemble = RuleEnsemble(model=model, model_type=self.model_type, feature_names=feature_names)

        # concatenate original features and rules
        X_rules = self.rule_ensemble.transform(X)
        
        # standardise linear variables
        if self.include_linear_features:
            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn=self.friedscale.scale(X)
            else:
                X_regn=X.copy()            
        
        # prepare training data
        if self.include_linear_features:
            X_concat = np.concatenate((X_rules, X_regn), axis=1)
        else:
            X_concat = X_rules

        # fit Lasso
        if self.mode == 'regression':
            if self.Cs is None: 
                # use defaultshasattr(self.Cs, "__len__"):
                n_alphas= 100
                alphas=None
            elif hasattr(self.Cs, "__len__"):
                n_alphas= None
                alphas=1./self.Cs
            else:
                n_alphas= self.Cs
                alphas=None
            self.lscv = LassoCV(n_alphas=n_alphas,alphas=alphas,cv=self.cv,random_state=self.random_state)
            self.lscv.fit(X_concat, y)
            self.coef_=self.lscv.coef_
            self.intercept_=self.lscv.intercept_
        else:
            Cs=10 if self.Cs is None else self.Cs
            self.lscv=LogisticRegressionCV(Cs=Cs,cv=self.cv,penalty='l1',random_state=self.random_state,solver='liblinear')
            self.lscv.fit(X_concat, y)
            self.coef_=self.lscv.coef_[0]
            self.intercept_=self.lscv.intercept_[0]
        return self

    def predict(self, X):
        """Predict outcome for X"""
        X_concat=np.zeros([X.shape[0],0])
        if self.include_linear_features:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat, self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
    
        rule_coefs = self.coef_[-len(self.rule_ensemble.rules):] 
        if len(rule_coefs)>0:
            X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.

        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.

        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=False):
        """Return the estimated rules

        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.

        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.coef_[i]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.coef_[i]
            output_rules += [(self.feature_names[i], 'linear',coef, 1)]
        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.coef_[i + n_features]
            output_rules += [(rule.__str__(), 'rule', coef,  rule.support)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules
 