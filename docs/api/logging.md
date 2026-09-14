---
title: Logging
---

Klax logging is built around [`Metrics`][klax.Metric] that are evaluated during 
training and whose outputs are recorded in the [`History`][klax.History], which
itself is contained in the [`TrainingContext`][klax.TrainingContext]. 
Metrics are typically, but not necessarily, based on a [`Loss`][klax.Loss]
objects, where the [`LossMetric`][klax.LossMetric] conveniently combines a loss
function and a dataset into a metric.
Thereby, a metric can be any callable that receives the current 
[`TrainingState`][klax.TrainingState] and returns a dictionary, mapping
strings, usually the metric names, to JAX arrays. 
The [`MetricLogger`][klax.MetricLogger] callback evaluates metrics 
every `log_every` number of steps and records them in the [`History`][klax.History]. 
Meanwhile, the [`ProgressMeter`][klax.ProgressMeter] callback retrieves the 
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
            - keys
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
