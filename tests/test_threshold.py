import pandas as pd
import numpy as np
from mapper import Matcher, assign_test

def test_assignment_boundary():
    x = np.arange(5, dtype=float)
    train = pd.DataFrame({"x": x, "y1": x*0 + 0, "y2": x*0 + 1, "y3": x*0 + 2, "y4": x*0 + 3})
    ideal = pd.DataFrame({"x": x, "y1": x*0 + 0, "y2": x*0 + 1, "y3": x*0 + 2, "y4": x*0 + 3,
                          "y5": x*0 + 4, "y6": x*0 + 5, "y7": x*0 + 6, "y8": x*0 + 7, "y9": x*0 + 8,
                          "y10": x*0 + 9, "y11": x*0 + 10, "y12": x*0 + 11, "y13": x*0 + 12, "y14": x*0 + 13,
                          "y15": x*0 + 14, "y16": x*0 + 15, "y17": x*0 + 16, "y18": x*0 + 17, "y19": x*0 + 18,
                          "y20": x*0 + 19, "y21": x*0 + 20, "y22": x*0 + 21, "y23": x*0 + 22, "y24": x*0 + 23,
                          "y25": x*0 + 24, "y26": x*0 + 25, "y27": x*0 + 26, "y28": x*0 + 27, "y29": x*0 + 28,
                          "y30": x*0 + 29, "y31": x*0 + 30, "y32": x*0 + 31, "y33": x*0 + 32, "y34": x*0 + 33,
                          "y35": x*0 + 34, "y36": x*0 + 35, "y37": x*0 + 36, "y38": x*0 + 37, "y39": x*0 + 38,
                          "y40": x*0 + 39, "y41": x*0 + 40, "y42": x*0 + 41, "y43": x*0 + 42, "y44": x*0 + 43,
                          "y45": x*0 + 44, "y46": x*0 + 45, "y47": x*0 + 46, "y48": x*0 + 47, "y49": x*0 + 48,
                          "y50": x*0 + 49})
    m = Matcher(train, ideal)
    matches = m.select_best_ideals()
    test = pd.DataFrame({"x": [0.0], "y": [0.0]})
    assigns = assign_test(ideal, matches, test)
    assert len(assigns) == 1
    assert assigns[0].accepted is True
    assert assigns[0].assigned_series in {"y1","y2","y3","y4"}
