# Contributing

Thank you for helping make this lab more useful to learners.

## Good contributions

- reproduce an existing experiment on different Apple Silicon hardware;
- clarify an explanation without hiding an important trade-off;
- fix a deterministic setup, preparation or evaluation issue;
- add a controlled experiment that changes one named factor; or
- improve accessibility, keyboard navigation or mobile behavior in the
  interactive guide.

## Before opening a pull request

1. Create an isolated Python 3.11 environment.
2. Do not commit model weights, adapters, dataset rows, caches or credentials.
3. Record model/dataset revisions, seeds, package versions and hardware.
4. Keep train, validation and test data separate.
5. Compare against an existing baseline on the same evaluation set.
6. Preserve failed-run evidence when it explains a decision.
7. Run:

```bash
python3 scripts/check_public_repo.py
python3 -m compileall -q scripts
node --check learning/app.js
python3 scripts/serve_learning_lab.py --check
```

## Reporting results

Include the complete configuration, correct/total count, accuracy, macro F1,
invalid-label count, peak memory and elapsed time. Clearly label results that
use a different model, dataset split, prompt, batch size or decoding strategy.

Do not describe a lower training loss as improved test performance. Report the
held-out evaluation.

## Data and privacy

Use public, licensed or explicitly authorized data. Never add real customer,
account, payment or authentication information to examples, logs or issues.
