---
title: Logging
---

Klax logging is built around metrics that are evaluated during training and
stored in a [History][klax.History]. A metric is any callable with a `name`
that receives the current [TrainingContext][klax.TrainingContext]. The default
[MetricLogger][klax.MetricLogger] callback evaluates metrics every `log_every`
steps, records them in history, and optionally prints progress (or a progress
bar). This is the default mechanism used by [klax.fit][] when `make_logger=True`.

::: klax.Metric
    options:
        members: false

::: klax.metric

::: klax.BatchMetric
    options:
        members:
            - __init__
            - __call__
---

::: klax.History
    options:
        members:
            - append
            - extend
            - keys
            - plot
            - save
            - load
---

::: klax.MetricLogger
    options:
        members:
            - __init__
            - add_metric