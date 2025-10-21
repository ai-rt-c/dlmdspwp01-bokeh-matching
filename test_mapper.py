import pandas as pd
import numpy as np
from mapper import Matcher

def test_argmin_selection_simple():
    # Toy grid
    x = np.arange(5)
    train = pd.DataFrame({
        "x": x, "y1": x*0 + 1, "y2": x*0 + 2, "y3": x*0 + 3, "y4": x*0 + 4
    })
    ideal = pd.DataFrame({
        "x": x, "y1": x*0 + 1.1, "y2": x*0 + 2.0, "y3": x*0 + 2.9, "y4": x*0 + 10,
        "y5": x*0 + 3.0, "y6": x*0 + 4.0, "y7": x*0 + 5.0, "y8": x*0 + 6.0, "y9": x*0 + 7.0,
        "y10": x*0 + 8.0, "y11": x*0 + 9.0, "y12": x*0 + 10.0, "y13": x*0 + 11.0, "y14": x*0 + 12.0,
        "y15": x*0 + 13.0, "y16": x*0 + 14.0, "y17": x*0 + 15.0, "y18": x*0 + 16.0, "y19": x*0 + 17.0,
        "y20": x*0 + 18.0, "y21": x*0 + 19.0, "y22": x*0 + 20.0, "y23": x*0 + 21.0, "y24": x*0 + 22.0,
        "y25": x*0 + 23.0, "y26": x*0 + 24.0, "y27": x*0 + 25.0, "y28": x*0 + 26.0, "y29": x*0 + 27.0,
        "y30": x*0 + 28.0, "y31": x*0 + 29.0, "y32": x*0 + 30.0, "y33": x*0 + 31.0, "y34": x*0 + 32.0,
        "y35": x*0 + 33.0, "y36": x*0 + 34.0, "y37": x*0 + 35.0, "y38": x*0 + 36.0, "y39": x*0 + 37.0,
        "y40": x*0 + 38.0, "y41": x*0 + 39.0, "y42": x*0 + 40.0, "y43": x*0 + 41.0, "y44": x*0 + 42.0,
        "y45": x*0 + 43.0, "y46": x*0 + 44.0, "y47": x*0 + 45.0, "y48": x*0 + 46.0, "y49": x*0 + 47.0,
        "y50": x*0 + 48.0
    })
    m = Matcher(train, ideal)
    matches = m.select_best_ideals()
    ideals = {mm.train_series: mm.ideal_series for mm in matches}
    assert ideals["y2"] == "y2"
    assert ideals["y3"] in {"y3","y5"}  # small ambiguity tolerated
