---
title: Callbacks
---

::: klax.CallbackArgs
    options:
        members:
            - loss
            - model
            - opt_state
            - val_loss
            - __init__
            - update

---

::: klax.Callback
    options:
        members:
            - __call__
            - on_training_start
            - on_training_end

---

::: klax.HistoryCallback
    options:
        members:
            - __init__
            - __call__
            - load
            - save 
            - plot
            - on_training_start
            - on_training_end