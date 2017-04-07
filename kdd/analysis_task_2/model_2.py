# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from build_model import AbstractModel


class RF_Model(AbstractModel):

    def build_model(self):
        pipeline = Pipeline([
            ('clf', RandomForestRegressor())
        ])
        parameters = {
            'clf__n_estimators': (5, 8, 10, 13, 15, 20),
            'clf__max_features': (5, 7, 10, 12, 14, 15),
            'clf__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15),
            'clf__min_samples_split': (2, 3, 5),
            'clf__min_samples_leaf': (1, 2, 5),
            # 'clf__min_weight_fraction_leaf': (),
            # 'clf__max_leaf_nodes': (),
            # 'clf__min_impurity_split': (),
            'clf__oob_score': (False, True),
        }
        self.parameters = parameters
        grid_search = GridSearchCV(pipeline, parameters, verbose=1)
        # model = RandomForestRegressor(
        #     n_estimators=15,
        #     criterion='mae',
        #     max_features=14,
        #     max_leaf_nodes=10
        # )
        # return model
        return grid_search


class RF_M(AbstractModel):
    def build_model(self):
        model = RandomForestRegressor(
            n_estimators=5,
            criterion='mae',
            max_features=7,
            min_samples_leaf=5,
            min_samples_split=2,
            max_leaf_nodes=10,
            max_depth=5,
            oob_score=True,
        )
        return model


class Gbdt(AbstractModel):
    def build_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            loss='lad',
            learning_rate=0.005,
            n_estimators=3000,
            max_depth=3,
            criterion='mae',
            min_samples_leaf=1,
            warm_start=True,
            random_state=0,
        )
        return model


class MLP(AbstractModel):
    def build_model(self):
        from sklearn.neural_network import MLPRegressor
