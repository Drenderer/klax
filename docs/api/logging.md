---
title: Logging
---

Klax logging is built around [`Metrics`][klax.Metric] that are evaluated during 
training and the resulting values are recorded in the [`History`][klax.History] 
of the [`TrainingContext`][klax.TrainingContext]. 
Typically, metrics are based on a [`Loss` function][klax.Loss]. The 
[`LossMetric`][klax.LossMetric] is designed to combine a loss function and 
a dataset into a metric.
A metric is any callable that receives the current 
[`TrainingState`][klax.TrainingState] and returns a dictionary mapping
strings (usually the metric names) to JAX arrays. 
The default [`MetricLogger`][klax.MetricLogger] callback evaluates metrics 
every `log_every` steps and records them in the history. 
The default [`ProgressMeter`][klax.ProgressMeter] callback retrieves the 
latest metric values from the history and prints them either directly or via 
a `tqdm` progress bar.

::: klax.Metric
    options:
        members:
            - __call__

::: klax.LossMetric
    options:
        members:
            - __init__
            - __call__

---

::: klax.History
    options:
        members:
            - append
            - save
            - load
            - plot

---

::: klax.MetricLogger
    options:
        members:
            - __init__

::: klax.ProgressMeter
    options:
        members:
            - __init__