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
from io import StringIO
import json
import re
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.linear_model import Lasso, LogisticRegression
from functools import reduce
from typing import List, Tuple


__all__ = ["RuleFit"]


class Winsorizer:
    """Performs Winsorization 1->1*
    Warning: this class should not be used directly.
    """

    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.winsor_lims = None

    def train(self, X):
        # get winsor limits
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(
            X > self.winsor_lims[1, :],
            np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
            np.where(
                X < self.winsor_lims[0, :],
                np.tile(self.winsor_lims[0, :], [X.shape[0], 1]),
                X,
            ),
        )
        return X_


class FriedScale:
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
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]
        # get multipliers
        X_trimmed = self.trim(X)
        scale_multipliers = np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals = len(np.unique(X[:, i_col]))
            if (
                num_uniq_vals > 2
            ):  # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col] = 0.4 / (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X, winsorize=True):
        if winsorize:
            return self.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(
            X > self.winsor_lims[1, :],
            np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
            np.where(
                X < self.winsor_lims[0, :],
                np.tile(self.winsor_lims[0, :], [X.shape[0], 1]),
                X,
            ),
        )
        return X_


class RuleCondition:
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(
        self,
        feature_index,
        threshold,
        operator,
        support,
        include_na=False,
        feature_name=None,
    ):
        if operator not in ('<', '>=', '<=', '>', '==', '!='):
            raise ValueError('Operator {} is not supported.'.format(operator))
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.include_na = include_na
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        feature = self.feature_name or 'feature_{}'.format(self.feature_index)

        if self.include_na:
            na_info = ' or %s is null' % feature
        else:
            na_info = ''
        
        return "%s %s %s%s" % (feature, self.operator, self.threshold, na_info)

    @classmethod
    def from_string(cls, s: str):
        """ Create a RuleCondition from a string, which should match the following syntax:
            feature_{i} (<=|>|==|!=) threshold (or feature_{i} is null| not null)
            Make sure there's no space in threshold
        """
        cond, *na_info = s.split('or')

        include_na = False
        if na_info and 'is null' in na_info[0]:
            include_na = True

        feature_idx, operator, threshold, *kwargs = cond.split()

        if operator in ('==', '!='):
            threshold = [float(i.strip()) for i in threshold[1:-1].split(",")]
        else:
            threshold = float(threshold)

        if feature_idx.startswith('feature_'):
            feature_index, feature_name = int(feature_idx.split('_')[1]), None
        else:
            feature_index, feature_name = None, feature_idx

        return cls(feature_index=feature_index, 
                    threshold=threshold,
                    operator=operator,
                    include_na=include_na,
                    support=0,
                    feature_name=feature_name)

    def transform(self, X):
        with warnings.catch_warnings():
            # Suppress RuntimeWarnings from NaN values
            warnings.filterwarnings("ignore")

            if self.operator == "<=":
                res = 1 * (X[:, self.feature_index] <= self.threshold)
            elif self.operator == ">":
                res = 1 * (X[:, self.feature_index] > self.threshold)
            elif self.operator == "==":
                res = 1 * np.array(
                    [i in self.threshold for i in X[:, self.feature_index]]
                )
            elif self.operator == "!=":
                res = 1 * np.array(
                    [i not in self.threshold for i in X[:, self.feature_index]]
                )
            else:
                raise ValueError("{} is not a valid operator".format(self.operator))

            if self.include_na:
                na_ind = pd.isnull(X[:, self.feature_index])
                res[na_ind] = 1
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(
            (
                self.feature_index,
                tuple(self.threshold)
                if isinstance(self.threshold, list)
                else self.threshold,
                self.operator,
                self.feature_name,
            )
        )


class Rule:
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self, rule_conditions):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        # self.prediction_value=prediction_value
        # self.rule_direction=None

    @classmethod
    def from_string(cls, s: str):
        """ Parse a rule from string, each RuleCondition is seperated by `&` """
        rules = [r.strip() for r in s.split('&')]
        rules = [RuleCondition.from_string(r) for r in rules]
        return cls(rules)

    def update_feature_information(self, feature_names):
        """ Update the `feature_name` or `feature_index` attribute for each RuleConditaion
            if only one of them is provided
         """
        conditions = []
        for cond in self.conditions:
            if cond.feature_index is not None and cond.feature_name is None:
                cond.feature_name = feature_names[cond.feature_index]
            if cond.feature_name is not None and cond.feature_index is None:
                cond.feature_index = list(feature_names).index(cond.feature_name)
            conditions.append(cond)
        self.conditions = set(conditions)

    def transform(self, X):
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __len__(self):
        return len(self.conditions)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # TODO: sum of hash values looks kinda sketchy
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


_OPERATORS = {"<=": ">", 
              ">": "<=", 
              "!=": "==", 
              "==": "!=",
              "<": ">=",
              ">=": "<"}


def get_opposite_operator(op):
    return _OPERATORS[op]


def extract_rules_from_scikit_tree(tree, feature_names=None):
    """ Extract a scikit-learn DecisionTree object into a set of Rule"""
    if hasattr(tree, "tree_"):
        tree = tree.tree_

    rules = set()

    def traverse_nodes(
        node_id=0, operator=None, threshold=None, feature=None, conditions=[]
    ):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(
                feature_index=feature,
                threshold=threshold,
                operator=operator,
                support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                feature_name=feature_name,
            )
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
        else:  # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions)
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()
    return rules


def extract_rules_from_xgb_tree(tree: dict, feature_names: List[str]):
    """ Extract a set of Rule from a single tree of xgb booster
        Note that support calculation is not supported for xgb, since the 
        parsed tree object doesn't contain number of samples 
    """
    rules = set()

    cache = [(tree, [])]  # a list of (node, rule conditions)
    while cache:
        node, conditions = cache.pop()

        if 'leaf' in node:
            # reaching a leaf node
            rules.add(Rule(conditions))
        else:
            feature_name = node['split']
            # when feature name is in the form of f1
            # 1 is the feature index, note in xgb the index starts from 1
            if re.findall(r'^f[0-9]+', feature_name):
                feature_index = int(feature_name[1:]) - 1
                feature_name = feature_names[feature_index]
            else:
                feature_index = feature_names.index(feature_name)

            threshold = node['split_condition']
            left_node_id, right_node_id = node['yes'], node['no']
            missing_node_id = node['missing']

            for child in node['children']:
                if child['nodeid'] == left_node_id:
                    rule_condition = RuleCondition(
                        feature_index=feature_index,
                        threshold=threshold,
                        operator='<',
                        support=0,
                        include_na=(child['nodeid']==missing_node_id),
                        feature_name=feature_name,
                        ) 
                else:
                    rule_condition = RuleCondition(
                        feature_index=feature_index,
                        threshold=threshold,
                        operator='>=',
                        support=0,
                        include_na=(child['nodeid']==missing_node_id),
                        feature_name=feature_name,
                        )
                cache.append((child, conditions+[rule_condition])) 
    return rules


def extract_rules_from_lgbm_tree(tree: dict, feature_names=None):
    """ Extract a set of Rule from a tree from a lightgbm booster
        the tree is one of the tree object from booster.dump_model()['tree_info']
    """
    if "tree_structure" in tree:
        tree = tree["tree_structure"]

    rules = set()
    n_total_sample = tree["internal_count"]

    def traverse_nodes(
        tree,
        operator=None,
        threshold=None,
        feature=None,
        n_sample=None,
        include_na=False,
        conditions=[],
    ):
        if tree.get("split_index", None) == 0:
            new_conditions = []
        else:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature

            rule_condition = RuleCondition(
                feature_index=feature,
                threshold=threshold,
                operator=operator,
                support=n_sample / n_total_sample,
                include_na=include_na,
                feature_name=feature_name,
            )
            new_conditions = conditions + [rule_condition]

        ## if not terminal node
        if "leaf_index" not in tree:
            feature = tree["split_feature"]
            threshold = tree["threshold"]
            operator = tree["decision_type"]
            if operator == "==":
                threshold = [float(i.strip()) for i in threshold.split("||")]
            else:
                threshold = float(threshold)
            n_sample = tree["internal_count"]
            # na_direction = "left" if tree["default_left"] else "right"

            traverse_nodes(
                tree=tree["left_child"],
                operator=operator,
                threshold=threshold,
                feature=feature,
                n_sample=n_sample,
                include_na=tree["default_left"],
                conditions=new_conditions,
            )

            traverse_nodes(
                tree=tree["right_child"],
                operator=get_opposite_operator(operator),
                threshold=threshold,
                feature=feature,
                n_sample=n_sample,
                include_na=(not tree['default_left']),
                conditions=new_conditions,
            )

        else:  # a leaf node
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions)
                # set the support attribute
                new_rule.support = tree['leaf_count'] / n_total_sample
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes(tree)
    return rules


class RuleEnsemble:
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

    def __init__(self, model, model_type="lightgbm", feature_names=None):
        if model_type.lower() not in ("tree", "forest", "gbdt", "xgb", "xgboost", "lightgbm"):
            raise ValueError(
                "Only supported model types are: {}".format(
                    ["tree", "forest", "gbdt", "xgb", "lightgbm"]
                )
            )

        if model_type.lower() in ("xgb", "xgboost") and feature_names is None:
            raise ValueError('`feature_names` must be provided when parsing xgb model. '
                            'Which should be in the same order as in fit process.')

        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names
        self.rules = set()
        if self.model is not None:
            ## TODO: Move this out of __init__
            self._extract_rules()
            # self.rules = list(self.rules)

    def __len__(self):
        return len(self.rules)

    def _extract_rules(self):
        """ Recursively extract rules from the model """
        if self.model_type == "tree":
            # a list of decision tree
            for tree in self.model:
                rules = extract_rules_from_scikit_tree(
                    tree, feature_names=self.feature_names
                )
                self.rules.update(rules)

        elif self.model_type in ("forest", "gbdt"):
            for tree in self.model.estimators_:
                rules = extract_rules_from_scikit_tree(
                    tree, feature_names=self.feature_names
                )
                self.rules.update(rules)

        elif self.model_type == "lightgbm":
            if hasattr(self.model, "booster_"):
                model = self.model.booster_
                for tree in model.dump_model()["tree_info"]:
                    rules = extract_rules_from_lgbm_tree(
                        tree, feature_names=self.feature_names
                    )
                    self.rules.update(rules)

        elif self.model_type in ('xgb', 'xgboost'):
            if hasattr(self.model, 'get_booster'):
                model = self.model.get_booster()
                with StringIO() as s:
                    model.dump_model(s, dump_format='json')
                    trees = json.loads(s.getvalue())

                for tree in trees:
                    rules = extract_rules_from_xgb_tree(
                        tree, feature_names=self.feature_names
                    )
                    self.rules.update(rules)

    def add_rule(self, rule):
        if isinstance(rule, str):
            rule = Rule.from_string(rule)
        
        if self.feature_names is not None:
            rule.update_feature_information(self.feature_names)
        
        self.rules.add(rule)

    def add_rules(self, rules):
        for rule in rules:
            self.add_rule(rule)

    def filter_rules(self, func):
        self.rules = set(filter(lambda x: func(x), self.rules))

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
        if coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else:  # else use the coefs to filter the rules we bother to interpret
            res = np.array(
                [
                    rule_list[i_rule].transform(X)
                    for i_rule in np.arange(len(rule_list))
                    if coefs[i_rule] != 0
                ]
            ).T
            res_ = np.zeros([X.shape[0], len(rule_list)])
            res_[:, coefs != 0] = res
            return res_

    def __str__(self):
        return '\n'.join(map(lambda x: x.__str__(), self.rules))


class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True, 
                        this will be the mean number of terminal nodes. 
                        Note that for xgb models, `max_depth` is set based on tree_size in the form of
                        2 ** `max_depth` <= `tree_size`
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
        fit_lr:         Boolean: Whether to fit a Logistic Regression model on the transformed dataset
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

    def __init__(
        self,
        tree_size=8,
        categorical_cols=None,
        sample_fract=1,
        max_rules=2000,
        learning_rate=0.01,
        model_type="lightgbm",
        mode="classification",
        lin_trim_quantile=0.025,
        lin_standardise=True,
        exp_rand_tree_size=True,
        fit_lr=True,
        include_linear_features=True,
        fit_with_cv=False,
        Cs=None,
        cv=3,
        verbose=0,
        n_jobs=1,
        random_state=1024,
    ):
        if model_type not in ("tree", "forest", "gbdt", "xgb", "xgboost", "lightgbm"):
            raise ValueError(
                "Supported model types are: {}.".format(["tree", "forest", "lightgbm"])
            )

        if mode not in ("classification", "regression"):
            raise ValueError("Mode should be one of classification and regression.")

        if mode == "lightgbm" and exp_rand_tree_size:
            warnings.warn(
                "Randomized tree depth is not supported when using LGBM as rule generator. "
                "Using constant depth instead."
            )

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
        self.fit_lr = fit_lr
        self.fit_with_cv = fit_with_cv
        self.cv = cv
        self.Cs = Cs
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    @staticmethod
    def find_max_depth(tree_size):
        """ Find the maximum depth for XGB models given maximum number of tree nodes """
        i = 1
        while 1:
            if 2 << i > tree_size:
                return i
            i += 1

    def _fit_scikit_estimator_with_warm_start(self, X, y, clf):
        """ Used to train RF and GBDT """
        N = X.shape[0]

        if not self.exp_rand_tree_size:
            # simply fit with constant tree size
            clf.fit(X, y)
        else:
            # randomise tree size as per Friedman 2005 Sec 3.3
            np.random.seed(self.random_state)
            tree_sizes = np.random.exponential(
                scale=self.tree_size - 2,
                size=int(np.ceil(self.max_rules * 2 / self.tree_size)),
            )
            tree_sizes = np.asarray(
                [2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],
                dtype=int,
            )
            i = int(len(tree_sizes) / 4)
            while np.sum(tree_sizes[0:i]) < self.max_rules:
                i = i + 1
            tree_sizes = tree_sizes[0:i]
            clf.set_params(warm_start=True)
            curr_est_ = 0
            for i_size in np.arange(len(tree_sizes)):
                size = tree_sizes[i_size]

                # warm_state=True seems to reset random_state, such that the trees are highly correlated,
                # unless we manually change the random_sate here.
                random_state = self.random_state if self.random_state else 0
                random_state += i_size

                clf.set_params(
                    **{
                        "n_estimators": curr_est_ + 1,
                        "max_leaf_nodes": size,
                        "random_state": random_state,
                    }
                )
                clf.fit(X, y)

                curr_est_ = curr_est_ + 1
            clf.set_params(warm_start=False)

        return clf

    def _fit_random_forest(self, X, y):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        N = X.shape[0]
        n_estimators = self.max_rules // self.tree_size
        sample_fract = self.sample_fract or min(0.5, (100 + 6 * np.sqrt(N)) / N)

        if self.mode == "classification":
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_leaf_nodes=self.tree_size,
                max_features=sample_fract,
                max_depth=100,
                random_state=self.random_state,
            )
        else:
            clf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_leaf_nodes=self.tree_size,
                subsample=sample_fract,
                max_depth=100,
                random_state=self.random_state,
            )

        return self._fit_scikit_estimator_with_warm_start(X, y, clf)

    def _fit_gbdt(self, X, y):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        N = X.shape[0]
        n_estimators = max(1, self.max_rules // self.tree_size)
        sample_fract = self.sample_fract or min(0.8, (100 + 6 * np.sqrt(N)) / N)

        if self.mode == "classification":
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.tree_size,
                subsample=sample_fract,
                max_depth=100,
                random_state=self.random_state,
            )
        else:
            clf = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.tree_size,
                subsample=sample_fract,
                max_depth=100,
                random_state=self.random_state,
            )

        return self._fit_scikit_estimator_with_warm_start(X, y, clf)

    def _fit_xgb(self, X, y):
        from xgboost import XGBClassifier, XGBRegressor

        N = X.shape[0]
        n_estimators = max(self.max_rules // self.tree_size, 1)
        sample_fract = self.sample_fract or min(0.8, (100 + 6 * np.sqrt(N)) / N)        
        max_depth = self.find_max_depth(self.tree_size)

        if self.mode == 'classification':
            model = XGBClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=self.learning_rate,
                subsample=sample_fract,
                random_state=self.random_state
                )
        else:
            model = XGBRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=self.learning_rate,
                subsample=sample_fract,
                random_state=self.random_state
                )
        model.fit(X, y)
        return model

    def _fit_lightgbm(self, X, y):
        from lightgbm import LGBMClassifier, LGBMRegressor

        N = X.shape[0]
        n_estimators = max(1, self.max_rules // self.tree_size)
        sample_fract = self.sample_fract or min(0.8, (100 + 6 * np.sqrt(N)) / N)

        if self.mode == "classification":
            model = LGBMClassifier(    
                num_leaves=self.tree_size,
                learning_rate=self.learning_rate,
                subsample=sample_fract,
                n_estimators=n_estimators,
                zero_as_missing=False,
                random_state=self.random_state
            )
        else:
            model = LGBMRegressor(
                num_leaves=self.tree_size,
                learning_rate=self.learning_rate,
                subsample=sample_fract,
                n_estimators=n_estimators,
                zero_as_missing=False,
                random_state=self.random_state
            )
        model.fit(
            X,
            y,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_cols or 'auto',
        )
        return model

    def add_rule(self, rule):
        """ Add a single rule, which will be combined with rules parsed from generators. """
        if hasattr(self, 'mannual_rules'):
            self.mannual_rules.add(rule)
        else:
            self.mannual_rules = set([rule])

    def add_rules(self, rules):
        """ Add a bunch of rules, which will be combined with rules parsed from generators. """
        if hasattr(self, 'mannual_rules'):
            self.mannual_rules.update(rules)
        else:
            self.mannual_rules = set(rules)

    def register_filter(self, filter):
        """ Register a filter function that will be called on the rules after fitting """
        if hasattr(self, 'rule_filter'):
            self.rule_filter.append(filter)
        else:
            self.rule_filter = [filter]

    def register_filters(self, filters):
        """ Register filter functions that will be called on the rules after fitting """
        if hasattr(self, 'rule_filter'):
            self.rule_filter.extend(filters)
        else:
            self.rule_filter = filters

    def fit(self, X, y=None, feature_names=None, prefitted_model=None):
        """ Fit and estimate linear combination of rule ensemble
            :param prefitted_model: A trained `tree` type model that will be used for generating features
        """
        # Enumerate features if feature names not provided
        N = X.shape[0]

        if hasattr(X, "columns"):
            # X is a DataFrame
            self.feature_names = feature_names = X.columns.tolist()
            X = X.copy().values

        if feature_names is None:
            self.feature_names = feature_names = ["feature_" + str(x) for x in range(0, X.shape[1])]
            

        # store the stdev of each feature for calculating the importance later on
        self.feature_stdev = np.std(X.astype('float'), axis=0)

        # build trees
        if prefitted_model is None:
            if self.model_type == "forest":
                model = self._fit_random_forest(X, y)
            elif self.model_type == "tree":
                # build a list of decision trees by training a RF
                model = self._fit_random_forest(X, y)
                model = model.estimators_
            elif self.model_type == "gbdt":
                model = self._fit_gbdt(X, y)
            elif self.model_type == "lightgbm":
                model = self._fit_lightgbm(X, y)
            elif self.model_type in ("xgb", "xgboost"):
                model = self._fit_xgb(X, y)
            self.model = model

            if self.verbose:
                print("{} training complete".format(self.model_type.capitalize()))
        else:
            self.model = model = prefitted_model

            if self.verbose:
                print('Using a prefitted model.')

        # extract rules
        self.rule_ensemble = RuleEnsemble(
            model=model, model_type=self.model_type, feature_names=feature_names
        )

        # add additional mannual specified rules if there's any
        if hasattr(self, 'mannual_rules'):
            if self.verbose:
                print('Add {} mannual rules'.format(len(self.mannual_rules)))
            self.rule_ensemble.add_rules(self.mannual_rules)

        # apply registered filters
        if hasattr(self, 'rule_filter'):
            if self.verbose:
                print('Apply {} registered filters.'.format(len(self.rule_filter)))
            for f in self.rule_filter:
                self.rule_ensemble.filter_rules(f)

        ########################################
        ## Fit a LR on the transformer dataset #
        ########################################

        if self.fit_lr:
            # concatenate original features and rules
            X_rules = self.rule_ensemble.transform(X)

            if self.verbose:
                print("{} rules were applied".format(len(self.rule_ensemble)))

            # standardise linear variables
            if self.include_linear_features:
                if self.lin_standardise:
                    self.friedscale.train(X)
                    X_regn = self.friedscale.scale(X)
                else:
                    X_regn = X.copy()

            # prepare training data, rule features are appended to the right
            if self.include_linear_features:
                X_concat = np.concatenate((X_regn, X_rules), axis=1)
            else:
                X_concat = X_rules

            if self.verbose:
                print("Start LR training.")

            # fit Lasso
            if self.mode == "regression":
                if self.fit_with_cv:
                    if self.Cs is None:
                        # use defaultshasattr(self.Cs, "__len__"):
                        n_alphas = 100
                        alphas = None
                    elif hasattr(self.Cs, "__len__"):
                        n_alphas = None
                        alphas = 1.0 / self.Cs
                    else:
                        n_alphas = self.Cs
                        alphas = None
                    self.lscv = LassoCV(
                        n_alphas=n_alphas,
                        alphas=alphas,
                        cv=self.cv,
                        verbose=self.verbose,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                else:
                    self.lscv = Lasso(verbose=self.verbose, random_state=self.random_state)
                self.lscv.fit(X_concat, y)
                self.coef_ = self.lscv.coef_
                self.intercept_ = self.lscv.intercept_
            else:
                if self.fit_with_cv:
                    Cs = 10 if self.Cs is None else self.Cs
                    self.lscv = LogisticRegressionCV(
                        Cs=Cs,
                        cv=self.cv,
                        penalty="l1",
                        verbose=self.verbose,
                        random_state=self.random_state,
                        solver="liblinear",
                        n_jobs=self.n_jobs,
                    )
                else:
                    self.lscv = LogisticRegression(
                        penalty="l1",
                        random_state=self.random_state,
                        verbose=self.verbose,
                        solver="liblinear",
                    )
                self.lscv.fit(X_concat, y)
                self.coef_ = self.lscv.coef_[0]
                self.intercept_ = self.lscv.intercept_[0]
        return self

    def predict(self, X):
        """Predict outcome for X"""
        X_concat = np.zeros([X.shape[0], 0])
        if self.include_linear_features:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat, self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat, X), axis=1)

        rule_coefs = self.coef_[-len(self.rule_ensemble.rules) :]
        if len(rule_coefs) > 0:
            X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
            if X_rules.shape[0] > 0:
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
        n_features = len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []

        ## Add coefficients for linear effects
        feature_stdev = self.feature_stdev
        for i in range(0, n_features):
            if self.lin_standardise:
                coef = self.coef_[i] * self.friedscale.scale_multipliers[i]
            else:
                coef = self.coef_[i]
            output_rules += [
                (
                    self.feature_names[i], 
                    "linear", 
                    coef, 
                    1, 
                    abs(coef) * feature_stdev[i]
                    )
                ]

        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef = self.coef_[i + n_features]
            output_rules += [
                (
                    rule,
                    "rule",
                    coef,
                    rule.support,
                    abs(coef) * rule.support * (1 - rule.support),
                )
            ]
        rules = pd.DataFrame(
            output_rules, columns=["rule", "type", "coef", "support", "importance"]
        )
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules
