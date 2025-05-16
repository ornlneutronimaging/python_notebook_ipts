import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    s = mo.ui.slider(1, 9)
    s

    return (s,)


if __name__ == "__main__":
    app.run()
