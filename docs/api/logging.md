---
title: Logging
---

While you can implement arbitrary logging functionality by creating a custom [Callback][klax.Callback], klax provides basic logging functionality with the [MetricLogger][klax.MetricLogger]. At it's core is the [Metric][klax.Metric] - a simple function, that maps a model to some value (or object). A basic and common metric is the [LossMetric][klax.LossMetric]. When called it, samples a batch from a dataset to evaluate the model on a [Loss function][klax.Loss]. 
The [MetricLogger][klax.MetricLogger] evaluates assigned [Metrics][klax.Metric] during training, and writes the resulting values to a dict-like [History][klax.History]. [Histories][klax.History] can be saved, loaded, combined and provide basic plotting functionality.

::: klax.Metric
    options:
        members: false

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