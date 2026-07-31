"""Corrected explanation-quality metrics for the alpha sweep.

IoU is not comparable across activation ranges: at density d the all-firing formula scores
IoU = d by construction, so a raw IoU only says how dense the neuron is. These four
quantities normalise that away.

    lift           = (|F & N| / |F|) / d          precision over base rate.
                     1.0 = the formula is statistically independent of the neuron.
                     ceiling is 1/d (perfect precision).
    normalised_fit = (lift - 1) / (1/d - 1)       lift rescaled to [0, 1].
                     0 = chance, 1 = the formula fires exactly where the neuron does.
    IoU_indep      = cov*d / (cov + d - cov*d)    the IoU a coverage-matched random
                     formula of the same size would score. The null for `best_iou`.
    iou_over_d     = IoU / d                      vs the all-fires baseline.
"""


def metrics(density, formula_cov, n_fires, n_inter, best_iou, n_fire_neuron=None):
    """All corrected metrics for one run. Returns None-valued dict on a timeout.

    Precision and recall are reported alongside lift because lift compresses them into one
    number: a formula can raise IoU by trading precision for recall (widening coverage),
    and lift will FALL while IoU rises. Only the pair shows that trade.
    """
    d, cov = density, formula_cov
    if None in (cov, n_fires, n_inter) or not n_fires or d <= 0:
        return {"precision": None, "recall": None, "lift": None, "normalised_fit": None,
                "iou_indep": None, "iou_over_d": None}

    precision = n_inter / n_fires
    recall = (n_inter / n_fire_neuron) if n_fire_neuron else None
    lift = precision / d
    # 1/d - 1 is the headroom above chance; it collapses as d -> 1, where no formula can
    # be informative because the neuron already fires everywhere.
    headroom = (1.0 / d) - 1.0
    nfit = (lift - 1.0) / headroom if headroom > 0 else float("nan")
    iou_indep = (cov * d) / (cov + d - cov * d) if (cov + d - cov * d) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4) if recall is not None else None,
        "lift": round(lift, 4),
        "normalised_fit": round(nfit, 4),
        "iou_indep": round(iou_indep, 4),
        "iou_over_d": round(best_iou / d, 4) if best_iou is not None else None,
    }
