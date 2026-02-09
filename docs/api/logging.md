---
title: Logging
---

While you can implement arbitrary logging functionality by creating a custom [Callback][klax.Callback], klax provides basic logging functionality with the [MetricLogger][klax.MetricLogger]. At it's core is the concept of a [Metric][klax.Metric]: A simple function, that maps a model to some value (or object). A basic and common metric is the [Evaluator][klax.Evaluator], which samples a batch from a dataset and evaluates the model on a function such as a [loss function][klax.Loss]. 
The [MetricLogger][klax.MetricLogger] keeps track of multiple [Metrics][klax.Metric] and evaluates them during training. The resulting metric values are written to a dict-like [history object][klax.History]. Besides storing the metrics, [histories][klax.History] provide basic functionality to save, load, combine and plot metrics.

::: klax.Metric
    options:
        members: false

::: klax.Evaluator
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