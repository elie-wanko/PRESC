from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# predict uncalibrated probabilities
def uncalibrated(x_train, x_test, y_train):
    # fit a model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # predict probabilities
    return model.decision_function(x_test)


# predict calibrated probabilities
def calibrated(x_train, x_test, y_train):
    # define model
    model = LogisticRegression()
    # define and fit calibration model
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(x_train, y_train)
    
    # predict probabilities
    return calibrated.predict_proba(x_test)[:, 1]
