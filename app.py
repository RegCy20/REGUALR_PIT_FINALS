from flask import Flask, render_template, request
from methods.adams_moulton import solve

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    values = {
        "expr": "y - x**2 + 1",
        "x0": "0",
        "y0": "0.5",
        "h": "0.2",
        "steps": "5",
        "corrections": "1",
    }

    if request.method == "POST":
        values = {
            "expr": request.form.get("expr", "").strip(),
            "x0": request.form.get("x0", "").strip(),
            "y0": request.form.get("y0", "").strip(),
            "h": request.form.get("h", "").strip(),
            "steps": request.form.get("steps", "").strip(),
            "corrections": request.form.get("corrections", "1").strip(),
        }

        try:
            results = solve(
                expr=values["expr"],
                x0=float(values["x0"]),
                y0=float(values["y0"]),
                h=float(values["h"]),
                steps=int(values["steps"]),
                corrections=int(values["corrections"]),
            )
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", results=results, error=error, values=values)

if __name__ == "__main__":
    app.run(debug=False)
