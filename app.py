from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            print("Form Values:")
            for field in [
                "closed_at_month", "closed_at_year", "is_ecommerce", 
                "age_last_milestone_year", "founded_at_month", 
                "age_first_milestone_year", "is_CA", 
                "age_last_funding_year", "is_MA", "tier_relationships"
            ]:
                print(field, request.form.get(field))
            
            data = CustomData(
                    closed_at_month=int(request.form.get("closed_at_month")),
                    closed_at_year=int(request.form.get("closed_at_year")),
                    is_ecommerce=int(request.form.get("is_ecommerce")),
                    age_last_milestone_year=float(request.form.get("age_last_milestone_year")),
                    founded_at_month=int(request.form.get("founded_at_month")),
                    age_first_milestone_year=float(request.form.get("age_first_milestone_year")),
                    is_CA=int(request.form.get("is_CA")),
                    age_last_funding_year=float(request.form.get("age_last_funding_year")),
                    is_MA=int(request.form.get("is_MA")),
                    tier_relationships=float(request.form.get("tier_relationships")),
                )

            data_df = data.get_data_as_df()

            predict_pipe = PredictPipeline()
            prediction = predict_pipe.predict(features=data_df)
         


            result = int(prediction[0])
            return render_template('index.html', results=result)

        except Exception as e:
            return render_template('index.html', results=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
