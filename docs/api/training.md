---
title: Calibration and Data Handling
---

::: klax.fit

---
## Data handling

::: klax.batch_data
::: klax.split_data


---
## Behind the scenes
Internally [`klax.fit`][klax.fit] assembles a 
[`TrainingState`][klax.TrainingState] and a 
[`TrainingContext`][klax.TrainingContext], sets up basic metrics and logging
and then runs the optimization via 
[`run_training_loop`][klax.run_training_loop].
These internals can be freely used to construct more complex training setups,
that go beyond the capabilities of [`klax.fit`][klax.fit]

::: klax.TrainingState
    options:
        members: false
::: klax.TrainingContext
    options:
        members: false
::: klax.run_training_loop
::: klax.StepFunction
    options:
        members:
            - __call__
::: klax.make_step
