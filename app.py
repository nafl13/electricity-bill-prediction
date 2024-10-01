from flask import Flask, request, render_template
import joblib
import numpy as np
import datetime as dt

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_tuned_model.pkl')

# Function to collect date-related inputs automatically


def get_date_features():
    now = dt.datetime.now()

    # Calculate PeriodOfDay based on current time
    period_of_day = get_period_of_day(now)

    return {
        'DayOfWeek': now.weekday(),
        'WeekOfYear': now.isocalendar()[1],
        'Day': now.day,
        'Month': now.month,
        'Year': now.year,
        'PeriodOfDay': period_of_day
    }

# Function to calculate period of the day (0 to 47)


def get_period_of_day(now):
    # Combine the hour and minute to calculate the period (0-47)
    hour = now.hour
    minute = now.minute
    period_of_day = (hour * 2) + (minute // 30)  # 0-47 (30-minute periods)
    return period_of_day

# Route for the homepage


@app.route('/')
def home():
    return render_template('index.html')

# Route to make predictions


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected holiday (just a gimmick, doesn't affect the prediction)
    selected_holiday = request.form['Holiday']

    # Set the HolidayFlag (1 if a holiday is selected, 0 otherwise)
    holiday_flag = 0 if selected_holiday == "None" else 1

    # Automatically collect date-related features
    date_features = get_date_features()

    # Get manual inputs from the form
    input_features = [
        float(request.form['ForecastWindProduction']),
        float(request.form['SystemLoadEA']),
        float(request.form['SMPEA']),
        float(request.form['ORKTemperature']),
        float(request.form['ORKWindspeed']),
        float(request.form['CO2Intensity']),
        float(request.form['ActualWindProduction']),
        float(request.form['SystemLoadEP2'])
    ]

    # Combine manual inputs with automatic date features (and include HolidayFlag)
    features = np.array([
        date_features['DayOfWeek'],
        date_features['WeekOfYear'],
        date_features['Day'],
        date_features['Month'],
        date_features['Year'],
        date_features['PeriodOfDay'],
        holiday_flag,  # Include the HolidayFlag in the feature array
        *input_features
    ]).reshape(1, -1)  # 15 features now

    # Make prediction
    prediction = model.predict(features)

    # Return prediction result
    return render_template('index.html', prediction_text=f'${prediction[0]:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
