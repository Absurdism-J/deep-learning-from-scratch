# Deep Learning from Scratch

Hand-typed implementations while working through 《深度学习入门》(斋藤康毅).
Building neural networks with pure NumPy — no frameworks, no shortcuts.

## Progress

- [x] Ch1: Python & NumPy basics
- [ ] Ch2: Perceptron — AND / NAND / OR / XOR implemented, theory wrap-up in progress
- [ ] Ch3: Neural network (forward propagation)
- [ ] Ch4: Loss functions & gradient
- [ ] Ch5: Backpropagation from scratch
- [ ] MNIST recognition ≥ 90% accuracy

## Structure

    ch01/   Python & NumPy basics
    ch02/   Perceptron gates (AND, NAND, OR) and XOR composed from them

## How to Run

    python ch02/perceptron_one.py   # each gate file runs its own truth-table self-test
    python ch02/perceptron-XOR.py   # XOR built from the three gates above

## Highlights

**XOR — the first "deep" moment.** A single perceptron can only draw one
straight line, so it can never separate XOR's four points. The fix is
composition: `XOR = AND(OR(x), NAND(x))` — three self-written gates wired
in two layers. What one layer cannot do, stacked layers can.

## Notes — pitfalls logged

- **Margin matters**: a decision boundary that sits exactly on data points
  misclassifies them. Good parameters keep distance from the data (→ SVM).
- **Module side effects**: top-level test code runs on import. Every module
  guards its self-test with `if __name__ == "__main__":` and passes its own
  truth table before joining a network.
- Module names must be valid Python identifiers — `perceptron-one.py` cannot
  be imported; use underscores.
