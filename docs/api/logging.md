---
title: Logging
---

Klax's logging is built around [`Metrics`][klax.Metric] that are evaluated 
during training and whose outputs are recorded in the [`History`][klax.History]
by a [`MetricLogger`][klax.MetricLogger] callback.
A metric computes a dictionary that maps strings (usually the metric names) to
arrays.
The [`LossMetric`][klax.LossMetric] conveniently combines a loss object and a 
dataset into a metric, that reports the loss's auxiliary output.
Meanwhile, the [`ProgressMeter`][klax.ProgressMeter] callback retrieves the 
latest metric values from the history and prints them either directly or via 
a `tqdm` progress bar.


## Metrics
::: klax.Metric
    options:
        members:
            - __init__
            - value
            - __call__

::: klax.metric

::: klax.LossMetric
    options:
        members:
            - __init__

---
## History

::: klax.History
    options:
        members:
            - append
            - keys
            - save
            - load
            - stack
            - plot

---
## Callbacks

::: klax.MetricLogger
    options:
        members:
            - __init__

::: klax.ProgressMeter
    options:
        members:
            - __init__
