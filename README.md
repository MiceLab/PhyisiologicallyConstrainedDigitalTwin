# Physiologically Constrained Digital Twin

Interactive Gradio explorer for the trained glycemia VAE decoders (safe and t1dexi).
Set the latent with the sliders, take a meal and bolus, and watch the generated blood glucose.
Doses go through the same plasma-insulin and Hovorka Ra models and the same input scaling that
built the training data.

```
pip install -r requirements.txt      # Python 3.10
python app.py --selftest             # checks the dose path end to end
APP_AUTH="user1:pass1,user2:pass2" python app.py
```

- `app.py`: the Gradio UI
- `twin.py`: the shipped models (`RUNS`), the physiological models, the input scaling and the decoder call
- `models/<run>/`: the shipped decoder checkpoint and its `training_config.json`

## Static version (GitHub Pages)

`docs/` is the same explorer running entirely in the browser, with no server and no TensorFlow:
`docs/twin.js` evaluates the decoder from weights exported to `docs/models/<dataset>.json`.
Publish it with Settings → Pages → Deploy from branch → `main` / `/docs`.

```
python export_web.py                 # re-export after changing RUNS or a checkpoint
node docs/selftest.mjs               # JS must reproduce the TF reference day
```

The exported weights are public on the site. Days with a moving latent (OU, resample) use
a different seeded RNG than the Python app, so they differ from it. Frozen-latent days match.

No patient data is included: only the decoder weights, their hyperparameters, and the
plasma-insulin / Ra scaling maxima. CHO, bolus and basal are entered by the user.
