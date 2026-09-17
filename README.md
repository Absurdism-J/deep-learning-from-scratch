# Deep Learning from Scratch

Hand-typed implementations while working through 《深度学习入门》(斋藤康毅).
Building neural networks with pure NumPy — no frameworks, no shortcuts.

## Progress

- [x] Ch1: Python & NumPy basics
- [x] Ch2: Perceptron — AND / NAND / OR, and XOR by composition
- [ ] Ch3: Neural network (forward propagation) — activation functions done
- [ ] Ch4: Loss functions & gradient
- [ ] Ch5: Backpropagation from scratch
- [ ] MNIST recognition ≥ 90% accuracy

## Structure

    ch01/   Python & NumPy basics
    ch02/   Perceptron gates (AND, NAND, OR) and XOR composed from them
    ch03/   Activation functions (step, sigmoid, ReLU) and forward propagation

## How to Run

    python ch02/perceptron_one.py   # each gate file runs its own truth-table self-test
    python ch02/perceptron-XOR.py   # XOR built from the three gates above
    python ch03/sigmoid.py          # activation functions with self-test

## Highlights

**XOR — the first "deep" moment.** A single perceptron can only draw one
straight line, so it can never separate XOR's four points. The fix is
composition: `XOR = AND(OR(x), NAND(x))` — three self-written gates wired
in two layers. What one layer cannot do, stacked layers can.

**Nonlinearity is the point.** Stacking linear layers still gives a linear
map — depth without nonlinear activation is an illusion. Step → sigmoid →
ReLU is the story of making activation both nonlinear and trainable.

## Notes — pitfalls logged

- **Margin matters**: a decision boundary that sits exactly on data points
  misclassifies them. Good parameters keep distance from the data (→ SVM).
- **Module side effects**: top-level test code runs on import. Every module
  guards its self-test with `if __name__ == "__main__":` and passes its own
  truth table before joining a network.
- Module names must be valid Python identifiers — `perceptron-one.py` cannot
  be imported; use underscores.
- **Sync discipline**: edit locally, push from local. If the web editor is
  ever used, `git pull` before the next push.