from flask import Flask, render_template, request
import neuralnet.predict as pr
app=Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    gpa = request.form["gpa"]
    score = request.form["test_score"]
    predictions=[]
    score=int(score)
    gpa=float(gpa)
    print(str(gpa)+", "+str(score))
    if score>36:
        predictions=pr.predict(gpa,score,"ACT")
    if score<=1600:
        predictions=pr.predict(gpa,score,"SAT1600")
    else:
        predictions = pr.predict(gpa,score,"SAT2400")
    if predictions[0]==1:
        return "I would recommend you apply as a match/safety."
    else:
        if predictions[0]==0:
            return "This would work as a high reach."
        return "Something went wrong."

@app.route('/')
def home():
    return render_template('website.html')

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')