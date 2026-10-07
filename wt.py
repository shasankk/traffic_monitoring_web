from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def get_weather(city):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()
    return "Weather data unavailable"

@app.route("/", methods=["GET", "POST"])
def index():
    weather = None
    city_name = None

    if request.method == "POST":
        city_name = request.form.get("city").strip()
        if city_name:
            weather = get_weather(city_name)

    return render_template("weather.html", weather=weather, city=city_name)

if __name__ == "__main__":
    app.run(debug=True)
